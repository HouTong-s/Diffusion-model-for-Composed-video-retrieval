from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

from src.model.blip import create_vit, init_tokenizer, load_checkpoint
from src.model.med import BertModel
from src.tools.utils import print_dist



class BLIPCir(nn.Module):
    def __init__(
        self,
        loss: Any,
        med_config="configs/med_config.json",
        image_size=384,
        vit="large",
        vit_grad_ckpt=True,
        vit_ckpt_layer=12,
        embed_dim=256,
        train_vit=False,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.loss = loss

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer
        )
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.train_vit = train_vit
        if not self.train_vit:
            # Do not train visual encoder
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        self.temp = 0.07

    def forward(self, batch, fabric, schedule_sampler=None, diffusion=None):
        ref_img, tar_feat, caption, _ = batch

        device = ref_img.device
        query_feat = self.get_text_image_fusion_feat(ref_img,caption)
                # Encode the target image
        tar_feat = tar_feat.to(device)
        tar_img_feat = F.normalize(tar_feat, dim=-1)

        loss, discrimination_loss, generation_loss = 0., 0., 0.

        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_feat = fabric.all_gather(query_feat, sync_grads=True)
            query_feat = einops.rearrange(query_feat, "d b e -> (d b) e")

            tar_img_feat = fabric.all_gather(tar_img_feat, sync_grads=True)
            tar_img_feat = einops.rearrange(tar_img_feat, "d b e -> (d b) e")

        discrimination_loss = self.loss(query_feat, tar_img_feat, self.temp)
        loss += discrimination_loss

        logit_scale = fabric.logit_scale

        t2v_logits = query_feat @ tar_feat.T
        v2t_logits = t2v_logits.T

        if self.stage == "discrimination":
            return loss, discrimination_loss, torch.zeros_like(discrimination_loss)

        t, weights = schedule_sampler.sample(query_feat.shape[0], query_feat.device)
        num = self.config.num

        query_feat, tar_feat = query_feat.detach(), tar_feat.detach()
        t2v_logits, v2t_logits = t2v_logits.detach(), v2t_logits.detach()
        a, b = query_feat.size(0), tar_feat.size(1)

        diagonal_mask = torch.eye(weights_t2v.size(0), weights_t2v.size(1)).bool().to(weights_t2v.device)
        weights_t2v = F.softmax(t2v_logits, dim=1)
        weights_v2t = F.softmax(v2t_logits, dim=1)
        weights_t2v.masked_fill_(diagonal_mask, -1)
        weights_v2t.masked_fill_(diagonal_mask, -1)


        target_embeds_neg = []
        for b in range(weights_t2v.size(0)):
            
            _, neg_idx = weights_t2v[b].topk(num, largest=True, sorted=True)
            temp = [tar_feat[b, :, :]]
            for i in neg_idx:
                temp.append(tar_feat[i, :, :])
            #temp (num+1,frames,feat_size)
            target_embeds_neg.append(torch.stack(temp, dim=0))
        target_embeds_neg = torch.stack(target_embeds_neg, dim=0)

        query_embeds_neg = []
        for b in range(weights_v2t.size(0)):
            _, neg_idx = weights_v2t[b].topk(num, largest=True, sorted=True)
            temp = [query_feat[b]]
            for i in neg_idx:
                temp.append(query_feat[i])
            #temp (num+1,feature_dim)
            query_embeds_neg.append(torch.stack(temp, dim=0))
        query_embeds_neg = torch.stack(query_embeds_neg, dim=0)

        pos = torch.ones((query_feat.size(0), 1), dtype=torch.float)
        if self.config.neg == 0:
            neg = torch.zeros((query_feat.size(0), num), dtype=torch.float)
        else:
            neg = -torch.ones((query_feat.size(0), num), dtype=torch.float)
        micro = torch.cat([pos, neg], dim=1).to(query_feat.device)

        output = diffusion.training_losses(self.diffusion_model, micro, t, {"text_emb": query_feat,
                                                                            "video_emb": target_embeds_neg},
                                            temp=self.config.d_temp)
        generation_loss += output["kl_loss"]

        output = diffusion.training_losses(self.diffusion_model_v, micro, t, {"text_emb": query_embeds_neg,
                                                                                "video_emb": tar_feat},
                                            temp=self.config.d_temp)
        generation_loss += output["kl_loss"]

        loss += generation_loss

        return loss
    
    def get_text_image_fusion_feat(self, ref_img, caption):
        device = ref_img.device
        if self.train_vit:
            ref_img_embs = self.visual_encoder(ref_img)
        else:
            with torch.no_grad():
                ref_img_embs = self.visual_encoder(ref_img)

        # Encode the reference image
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        # Shift encoder
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        query_embs = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        query_feat = query_embs.last_hidden_state[:, 0, :]
        query_feat = F.normalize(self.text_proj(query_feat), dim=-1)
        return query_feat

def blip_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model, msg = load_checkpoint(model, ckpt_path)
        print_dist("missing keys:")
        print_dist(msg.missing_keys)
    return model
