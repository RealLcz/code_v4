from torch import nn
import torch
from MoE import MoE
from graph_grasper import GraphGrasper
import torch.nn.functional as F


class QuestionEmbeder(nn.Module):
    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        # 冻结嵌入层
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, question_text):
        """
        ：question_text [B,] → 输出 [B, seq_len_q, d]
        """
        # 批量编码
        inputs = self.tokenizer(
            question_text,
            padding=True,
            truncation=True,
            max_length=self.args.max_question_len,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.model.device)

        with torch.no_grad():
            word_embeds = self.model.model.embed_tokens(inputs["input_ids"])  # [B, seq_len_q, d]
        return word_embeds

class CrossAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.dim_input
        assert args.moe_input_size == self.d_model

        # 批量交叉注意力层
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=self.d_model * 4,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=args.cross_attention_layers
        )

        # 批量MoE
        self.moe = MoE(
            input_size=self.d_model,
            hidden_size=args.moe_hidden_size,
            output_size=args.moe_output_size,
            num_experts=args.moe_num_experts,
            k=args.moe_k,
            noisy_gating=True
        )

    def forward(self, multimodal_emb, question_emb):
        """
        - multimodal_emb: [B, N, d]
        - question_emb: [B, seq_len_q, d]
        """
        B, N, d = multimodal_emb.shape

        tgt_mask = None

        cross_emb = self.transformer_decoder(
            tgt=multimodal_emb,
            memory=question_emb,
            tgt_mask=tgt_mask
        )

        cross_emb = cross_emb.reshape(B*N, d)

        # MoE：[B*N, d] → [B*N, d]
        final_prompt, moe_loss = self.moe(cross_emb)
        final_prompt = final_prompt.reshape(B, N, d)
        return final_prompt, moe_loss

class MyModel(nn.Module):
    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.tokenizer = tokenizer
        self.model = model
        self.graph_grasper = GraphGrasper(args, model, tokenizer).to(self.device)
        self.question_embeder = QuestionEmbeder(args, tokenizer, model).to(self.device)
        self.cross_modality_MHA = CrossAttention(args).to(self.device)


        self.classifier = nn.Sequential(
            nn.Linear(args.llm_emb_dim, args.llm_emb_dim // 2),
            nn.LayerNorm(args.llm_emb_dim // 2),
            nn.GELU(),
            nn.Linear(args.llm_emb_dim // 2, 2)
        ).to(self.device)

    def forward(self, gat_inputs, batch_pyg_graphs, batch_text_attributes, batch_triples, question_text, labels=None):
        B = len(batch_pyg_graphs)

        # [B, N, d]
        multimodal_emb, link_loss = self.graph_grasper(
            gat_inputs=gat_inputs,
            batch_pyg_graphs=batch_pyg_graphs,
            batch_text_attributes=batch_text_attributes,
            batch_triples=batch_triples
        )

        # [B, seq_len_q, d]
        question_emb = self.question_embeder(question_text)

        # [B, N, d]
        the_prompt, moe_loss = self.cross_modality_MHA(multimodal_emb, question_emb)

        # [B, seq_len_q + N, d]
        input_embeds = torch.cat([question_emb, the_prompt], dim=1)

        outputs = self.model(
            input_embeds=input_embeds,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]  # [B, seq_len_total, d]
        cls_hidden = hidden_states[:, -1, :]  # [B, d]
        logits = self.classifier(cls_hidden)  # [B, 2]

        total_loss = 0.0
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
            total_loss = cls_loss + self.args.link_loss_weight * link_loss + self.args.moe_loss_weight * moe_loss

        return outputs, logits, total_loss, link_loss, moe_loss