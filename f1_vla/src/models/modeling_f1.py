from dataclasses import dataclass
from typing import List
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from lerobot.policies.pi0.modeling_pi0 import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from f1_vla.src.models.paligemma_with_expert import (
    PaliGemmaWithExpertModel
)
from f1_vla.src.utils.model_utils import sample_with_top_k_top_p_


class F1FlowMatching(nn.Module):

    def __init__(self, config, patch_nums=None, vae=None, **kwargs):
        super().__init__()
        self.config = config

        self.paligemma_with_expert = PaliGemmaWithExpertModel(self.config)

        if self.config.use_world_model:
            self.vlm_expert_hidden_size = config.und_expert_config.hidden_size
            self.wm_expert_hidden_size = config.gen_expert_config.hidden_size

            # world model modules
            self.patch_nums = patch_nums
            self.vae_dim = config.gen_expert_config.vae.z_channels
            self.vocab_size = vae.V
            self.L = sum(pn ** 2 for pn in self.patch_nums)
            self.first_l = self.patch_nums[0] ** 2 + self.L - 1
            init_std = math.sqrt(1 / self.wm_expert_hidden_size / 3)
            self.num_stages_minus_1 = len(self.patch_nums) - 1

            # 1. temporal downsampling
            self.temporal_conv = TemporalDownsampling(
                self.vae_dim, 
                self.vae_dim, 
                config.gen_expert_config.temporal_conv_kernel_size, 
                config.gen_expert_config.temporal_conv_stride
            )

            # 2. wm_embedding
            self.wm_embeddings = nn.Linear(self.vae_dim, self.wm_expert_hidden_size)
            self.wm_hist_pos_embs = nn.Parameter(torch.empty(1, self.L - 1, self.vae_dim))
            nn.init.trunc_normal_(self.wm_hist_pos_embs.data, mean=0, std=init_std)
            self.wm_sos_token_embs = nn.Parameter(torch.empty(1, 1, self.wm_expert_hidden_size))
            nn.init.trunc_normal_(self.wm_sos_token_embs.data, mean=0, std=init_std)
            self.wm_cond_pos_embs = nn.Parameter(torch.empty(1, self.first_l, self.wm_expert_hidden_size))
            nn.init.trunc_normal_(self.wm_cond_pos_embs.data, mean=0, std=init_std)
            pos_1LC = []
            for pn in self.patch_nums:
                pe = torch.empty(1, pn*pn, self.wm_expert_hidden_size)
                nn.init.trunc_normal_(pe, mean=0, std=init_std)
                pos_1LC.append(pe)
            pos_1LC = torch.cat(pos_1LC, dim=1)
            self.wm_output_position_embedding = nn.Parameter(pos_1LC)  # abosulte position embedding

            # 3. position embedding
            d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
            dT = d.transpose(1, 2)
            lvl_1L = dT[:, 0].contiguous()
            self.register_buffer('lvl_1L', lvl_1L)
            self.lvl_embed = nn.Embedding(len(self.patch_nums), self.wm_expert_hidden_size)
            nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

            # 4. (Optional) Classifier free guidance
            self.cfg_scale = 0.0
            if self.cfg_scale > 0:
                self.register_buffer('cfg_embedding', torch.empty(1, 1, self.vlm_expert_hidden_size))
                nn.init.trunc_normal_(self.cfg_embedding, mean=0, std=init_std)
            
            # 5. mask ratio
            self.mask_method = "mixture"
            self.seq_mask_ratio = 0.1
            self.token_mask_ratio = 0.15
            if self.seq_mask_ratio > 0 or self.token_mask_ratio > 0:
                self.register_buffer('mask_embedding', torch.empty(1, 1, vae.Cvae))
                nn.init.trunc_normal_(self.mask_embedding, mean=0, std=init_std)

            # 6. projection
            self.wm_out_layer_norm = nn.LayerNorm(self.wm_expert_hidden_size)
            self.wm_out_proj = nn.Linear(self.config.proj_width, self.vocab_size)

            self.vae = vae
            self.num_resolutions = config.gen_expert_config.num_resolutions

        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        # replace the action_out_proj from a linear layer to a mlp
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        training_args = kwargs.get("training_args", None)

        self.set_requires_grad(training_args)

    def set_requires_grad(self, training_args=None):
        if training_args is None:
            self.freeze_vision_encoder = True
            self.freeze_gen_expert = False
            self.train_act_expert_only = False
            self.train_gen_expert_only = False
            self.train_state_proj = True
        else:
            self.freeze_vision_encoder = training_args.freeze_vision_encoder
            self.train_act_expert_only = training_args.train_act_expert_only
            self.train_gen_expert_only = training_args.train_gen_expert_only
            self.train_state_proj = training_args.train_state_proj
            self.freeze_gen_expert = training_args.freeze_gen_expert

        self.paligemma_with_expert.set_requires_grad(
            freeze_vision_encoder=self.freeze_vision_encoder,
            freeze_gen_expert=self.freeze_gen_expert,
            train_act_expert_only=self.train_act_expert_only,
            train_gen_expert_only=self.train_gen_expert_only,
        )
        for params in self.state_proj.parameters():
            params.requires_grad = self.train_state_proj

        if training_args.train_gen_expert_only:
            freeze_modules = ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]
            for name, param in self.named_parameters():
                if any (x in name for x in freeze_modules):
                    param.requires_grad = False

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        # state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_wm_inputs(self, world_model_input_embs, world_model_output_embs=None):
        """
        Construct the world model inputs, including:
            1. temporal downsampling + pos embedding: since the input is a sequence of images, 
                    we need to downsample the temporal dimension. We view the output as the condition
                    and add the condition position embedding.
            2. world model embedding + sos token + pos embedding: we embed the condition using a linear layer and 
                    add the sos token embedding. After that, we add the absolute position embedding.
            3. (Optional) teacher-forcing style learning: we add the ground truth world model embedding to the 
                    world model output embedding. Attention, it only works when training. As same as the previous
                    step, we add the position embedding and level embedding to the ground truth world model embedding.
                    Note, the position embedding add here is not related to the condition, so we pad them with zeros.
            4. attention mask: we set the block attention mask according to the scale.
        """
        # 1. temporal downsampling + pos embedding
        world_model_inputs = self.temporal_conv(world_model_input_embs)
        world_model_inputs = world_model_inputs.squeeze(1)
        world_model_inputs += self.wm_hist_pos_embs     # 679

        # 2. world embedding + sos token +pos embedding
        world_model_embs = self.wm_embeddings(world_model_inputs)
        wm_sos_token = self.wm_sos_token_embs.expand(world_model_embs.shape[0], 1, -1)
        world_model_embs = torch.cat([wm_sos_token, world_model_embs], dim=1)   # 1+679
        world_model_embs += self.wm_cond_pos_embs

        # 3. teacher-forcing style learning
        if world_model_output_embs is not None:
            if self.mask_method is not None and (self.seq_mask_ratio > 0 or self.token_mask_ratio > 0):
                batch_size, seq_len, hidden_dim = world_model_output_embs.shape
                device = world_model_output_embs.device
                if self.mask_method == "sequence":
                    mask_flags = (torch.rand(batch_size, device=device) < self.seq_mask_ratio).unsqueeze(1).unsqueeze(2)
                    mask_emb = self.mask_embedding.expand(batch_size, seq_len, -1)
                    world_model_output_embs = torch.where(mask_flags, mask_emb, world_model_output_embs)
                elif self.mask_method == "token":
                    mask_flags = (torch.rand(batch_size, seq_len, 1, device=device) < self.token_mask_ratio)
                    mask_emb = self.mask_embedding.expand(batch_size, seq_len, hidden_dim)
                    world_model_output_embs = torch.where(mask_flags, mask_emb, world_model_output_embs)
                elif self.mask_method == "mixture":
                    batch_mask_flags = (torch.rand(batch_size, 1, 1, device=device) < self.seq_mask_ratio)
                    token_mask_flags = (torch.rand(batch_size, seq_len, 1, device=device) < self.token_mask_ratio)
                    final_mask_flags = batch_mask_flags & token_mask_flags
                    mask_emb = self.mask_embedding.expand(batch_size, seq_len, hidden_dim)
                    world_model_output_embs = torch.where(final_mask_flags, mask_emb, world_model_output_embs)
                else:
                    raise ValueError(f"Invalid mask level: {self.mask_method}")

            gt_world_model_embs = self.wm_embeddings(world_model_output_embs)
            world_model_embs = torch.cat([world_model_embs, gt_world_model_embs], dim=1)
            padding_pos_embs = torch.zeros(1, self.first_l - 1, self.wm_expert_hidden_size).to(world_model_embs.device)
            wm_output_pos_embs = torch.cat([padding_pos_embs, self.wm_output_position_embedding], dim=1)
            padding_lvl_embs = torch.zeros(1, self.first_l - 1, self.wm_expert_hidden_size).to(world_model_embs.device)
            wm_output_lvl_embs = torch.cat([padding_lvl_embs, self.lvl_embed(self.lvl_1L)], dim=1)
            world_model_embs += wm_output_pos_embs + wm_output_lvl_embs

        embs = world_model_embs

        att_masks = [0] * (self.L - 1)
        for res_idx, pn in enumerate(self.patch_nums):
            att_masks += [1] + ([0] * (pn * pn - 1))
            # only use the first n resolution
            if res_idx == self.num_resolutions - 1:
                break
        att_masks = torch.tensor(att_masks, device=embs.device)
        att_masks = att_masks[None, :].expand(embs.shape[0], len(att_masks))

        pad_masks = torch.ones(embs.shape[0], att_masks.shape[1], dtype=torch.bool, device=embs.device)

        return embs, pad_masks, att_masks

    def forward_with_world_model(
        self, 
        images: List[torch.Tensor], 
        img_masks: List[torch.Tensor], 
        lang_tokens: torch.Tensor, 
        lang_masks: torch.Tensor, 
        state: torch.Tensor, 
        world_model_input_embs: List[torch.Tensor], 
        world_model_output_embs: List[torch.Tensor],
        actions: torch.Tensor, 
        noise: torch.Tensor | None = None, 
        time: torch.Tensor | None = None,
        top_k: int = 900,
        top_p: float = 0.95,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """use the world model to guide the acton generation
            step1: prepare the noise and time for the action generation
            step2: prepare the world model inputs
            step3: embed the prefix, world model inputs and suffix
            step4: forward the prefix, world model inputs and suffix
            step5: compute the loss
        """
        if self.train_gen_expert_only:
            gen_embs, gen_pad_masks, gen_att_masks = self.embed_wm_inputs(
                world_model_input_embs, world_model_output_embs
            )
            und_embs, und_pad_masks, und_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            if self.cfg_scale > 0:
                batch_size = und_embs.shape[0]
                seq_len = und_embs.shape[1]
                mask_flags = (torch.rand(batch_size, device=und_embs.device) < self.cfg_scale).unsqueeze(1).unsqueeze(2)
                cfg_emb = self.cfg_embedding.expand(batch_size, seq_len, -1)
                und_embs = torch.where(mask_flags, cfg_emb, und_embs)

            pad_masks = torch.cat([und_pad_masks, gen_pad_masks], dim=1)
            att_masks = torch.cat([und_att_masks, gen_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            (_, gen_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[und_embs, gen_embs, None],
                use_cache=False,
                fill_kv_cache=False,
            )
            gen_out = gen_out.to(dtype=torch.float32)
            gen_out = self.wm_out_proj(self.wm_out_layer_norm(gen_out))[:, -self.L:]

            return 0, gen_out

        else:
            bsize = state.shape[0]
            device = state.device

            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)

            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            # compute the position and level embeddings for the world model output
            gen_embs, gen_pad_masks, gen_att_masks, gen_pos_lvl_embs = self._preparse_world_model_inputs(
                world_model_input_embs, device
            )
            und_embs, und_pad_masks, und_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )

            # 2. prepare the mask and position ids
            pad_masks = torch.cat([und_pad_masks, gen_pad_masks], dim=1)
            att_masks = torch.cat([und_att_masks, gen_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            cur_position = und_embs.shape[1] + gen_embs.shape[1]

            # 3. compute KV cache
            (_, gen_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks[:, :cur_position, :cur_position],
                position_ids=position_ids[:, :cur_position],
                past_key_values=None,
                inputs_embeds=[und_embs, gen_embs, None],
                use_cache=self.config.use_cache,
                fill_kv_cache=True,
            )

            # 4. generate world model output
            all_gen_logits = []
            all_gen_indices = []
            cur_L = 0
            f_hat = torch.zeros(bsize, self.vae_dim, self.patch_nums[-1], self.patch_nums[-1]).to(device).to(gen_embs.dtype)
            for si, pn in enumerate(self.patch_nums):
                cur_L += pn * pn
                if si != 0:
                    x = next_token_map
                    start_idx = cur_position
                    end_idx = cur_position + pn * pn
                    (_, gen_out, _), past_key_values = self.paligemma_with_expert.forward(
                        attention_mask=att_2d_masks[:, start_idx: end_idx, :end_idx],   # bs, cur_len, prefix_len
                        position_ids=position_ids[:, start_idx: end_idx],                    # bs, cur_len
                        past_key_values=past_key_values,
                        inputs_embeds=[None, x, None],
                        use_cache=self.config.use_cache,
                        fill_kv_cache=False,
                        cat_past_key_values=True,
                    )
                    cur_position = end_idx

                gen_out = gen_out[:, -1:] if si == 0 else gen_out
                gen_out = gen_out.to(dtype=torch.float32)
                logits = self.wm_out_proj(self.wm_out_layer_norm(gen_out))
                all_gen_logits.append(logits.clone())

                gen_indices = sample_with_top_k_top_p_(
                    logits, rng=rng, top_k=top_k, top_p=top_p, num_samples=num_samples
                )[:, :, 0]
                all_gen_indices.append(gen_indices)

                h_BChw = self.vae.quantize.embedding(gen_indices)   # B, l, Cvae
                h_BChw = h_BChw.transpose_(1, 2).reshape(bsize, self.vae_dim, pn, pn)
                f_hat, next_token_map = self.vae.quantize.get_next_autoregressive_input(
                    si, len(self.patch_nums), f_hat, h_BChw
                )

                if si != self.num_stages_minus_1:   # prepare for next stage
                    next_token_map = next_token_map.view(bsize, self.vae_dim, -1).transpose(1, 2)
                    next_token_map = self.wm_embeddings(next_token_map) + \
                        gen_pos_lvl_embs[:, self.first_l - 1 + cur_L:cur_L + self.first_l - 1 + self.patch_nums[si+1] ** 2]

                if si == self.num_resolutions - 1:
                    break

            # 5. generate action
            act_embs, act_pad_masks, act_att_masks = self.embed_suffix(state, x_t, time)
            act_len = act_pad_masks.shape[1]

            prefix_pad_2d_masks = pad_masks[:, None, :].repeat(1, act_len, 1)
            act_att_2d_masks = make_att_2d_masks(act_pad_masks, act_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, act_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(act_pad_masks, dim=1) - 1

            (_, _, act_out), _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, None, act_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
            )
            act_out = act_out[:, -self.config.chunk_size :]
            # Original openpi code, upcast attention output
            act_out = act_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(act_out)

            losses = F.mse_loss(u_t, v_t, reduction="none")

            gen_logits = torch.cat(all_gen_logits, dim=1)
            gen_logits = gen_logits[:, -self.L:]

            return losses, gen_logits

    def _preparse_world_model_inputs(self, world_model_input_embs, device):
        # compute the position and level embeddings for the world model input
        padding_pos_embs = torch.zeros(1, self.first_l - 1, self.wm_expert_hidden_size).to(device)
        wm_output_pos_embs = torch.cat([padding_pos_embs, self.wm_output_position_embedding], dim=1)    # 679+680
        padding_lvl_embs = torch.zeros(1, self.first_l - 1, self.wm_expert_hidden_size).to(device)
        wm_output_lvl_embs = torch.cat([padding_lvl_embs, self.lvl_embed(self.lvl_1L)], dim=1)     # 679+681
        wm_pos_lvl_embs = wm_output_pos_embs + wm_output_lvl_embs

        wm_embs, wm_pad_masks, wm_att_masks = self.embed_wm_inputs(
            world_model_input_embs, world_model_output_embs=None
        )
        wm_embs = wm_embs + wm_pos_lvl_embs[:, :wm_embs.shape[1]]

        return wm_embs, wm_pad_masks, wm_att_masks, wm_pos_lvl_embs

    def sample_actions_with_world_model(
        self, 
        images: List[torch.Tensor], 
        image_masks: List[torch.Tensor], 
        lang_tokens: torch.Tensor, 
        lang_masks: torch.Tensor, 
        state: torch.Tensor, 
        world_model_input_embs: torch.Tensor, 
        predict_action_only: bool = True,
        noise = None,
        top_k: int = 900,
        top_p: float = 0.95,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        gen_embs, gen_pad_masks, gen_att_masks, gen_pos_lvl_embs = self._preparse_world_model_inputs(
            world_model_input_embs, device
        )
        und_embs, und_pad_masks, und_att_masks = self.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )

        # 2. prepare the mask and position ids
        pad_masks = torch.cat([und_pad_masks, gen_pad_masks], dim=1)
        att_masks = torch.cat([und_att_masks, gen_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        cur_position = und_embs.shape[1] + gen_embs.shape[1]

        # 3. compute KV cache
        (_, gen_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks[:, :cur_position, :cur_position],
            position_ids=position_ids[:, :cur_position],
            past_key_values=None,
            inputs_embeds=[und_embs, gen_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        # 4. generate world model output
        all_gen_logits = []
        all_gen_indices = []
        cur_L = 0
        f_hat = torch.zeros(bsize, self.vae_dim, self.patch_nums[-1], self.patch_nums[-1]).to(device).to(gen_embs.dtype)
        for si, pn in enumerate(self.patch_nums):
            cur_L += pn * pn
            if si != 0:
                x = next_token_map
                start_idx = cur_position
                end_idx = cur_position + pn * pn
                (_, gen_out, _), past_key_values = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks[:, start_idx: end_idx, :end_idx],   # bs, cur_len, prefix_len
                    position_ids=position_ids[:, start_idx: end_idx],                    # bs, cur_len
                    past_key_values=past_key_values,
                    inputs_embeds=[None, x, None],
                    use_cache=self.config.use_cache,
                    fill_kv_cache=False,
                    cat_past_key_values=True,
                )
                cur_position = end_idx

            gen_out = gen_out[:, -1:] if si == 0 else gen_out
            gen_out = gen_out.to(dtype=torch.float32)
            logits = self.wm_out_proj(self.wm_out_layer_norm(gen_out))
            all_gen_logits.append(logits.clone())

            gen_indices = sample_with_top_k_top_p_(
                logits, rng=rng, top_k=top_k, top_p=top_p, num_samples=num_samples
            )[:, :, 0]
            all_gen_indices.append(gen_indices)

            h_BChw = self.vae.quantize.embedding(gen_indices)   # B, l, Cvae
            h_BChw = h_BChw.transpose_(1, 2).reshape(bsize, self.vae_dim, pn, pn)
            f_hat, next_token_map = self.vae.quantize.get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw
            )

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(bsize, self.vae_dim, -1).transpose(1, 2)
                next_token_map = self.wm_embeddings(next_token_map) + \
                    gen_pos_lvl_embs[:, self.first_l - 1 + cur_L:cur_L + self.first_l - 1 + self.patch_nums[si+1] ** 2]

            if si == self.num_resolutions - 1:
                break

        # 5. generate action
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        if predict_action_only:
            return x_t

        # pred_imgs = self.vae.fhat_to_img(f_hat)

        return ActionOutput(
            actions=x_t, 
            pred_imgs=None, 
            all_logits=torch.cat(all_gen_logits, dim=1), 
            all_pred_img_indices=torch.cat(all_gen_indices, dim=1)
        )

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        act_embs, act_pad_masks, act_att_masks = self.embed_suffix(state, x_t, timestep)
        act_len = act_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].repeat(1, act_len, 1)

        act_att_2d_masks = make_att_2d_masks(act_pad_masks, act_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, act_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(act_pad_masks, dim=1) - 1

        inputs_embeds = [None, None, act_embs] if self.config.use_world_model else [None, act_embs]
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            is_eval=True,
        )
        suffix_out = outputs_embeds[-1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t


class TemporalDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=4, stride=4):
        super(TemporalDownsampling, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernal_size,
            stride=stride,
            padding=0
        )

    def forward(self, x):
        batch_size, timesteps, seq_len, dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * seq_len, timesteps, dim)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1).reshape(batch_size, seq_len, 1, dim)
        x = x.permute(0, 2, 1, 3)

        return x


@dataclass
class ActionOutput:
    actions: torch.Tensor
    pred_imgs: torch.Tensor | None = None
    all_logits: torch.Tensor | None = None
    all_pred_img_indices: torch.Tensor | None = None
