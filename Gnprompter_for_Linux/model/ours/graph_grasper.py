import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN import GAT
from data_processing.change_umls_into_needed import generate_pos_neg_triples

class RotatEDecoder(nn.Module):
    def __init__(self, args, num_rels, h_dim):
        super().__init__()
        self.args = args
        self.gamma = args.link_gamma
        self.num_rels = num_rels
        self.h_dim = h_dim
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim // 2))
        self.embedding_range = (self.gamma + 2.0) / h_dim
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)

    def score(self, entity_emb, head_idx, rel_idx, tail_idx):
        """
       ：entity_emb [B, N, d], head_idx/rel_idx/tail_idx [B, K]
        """
        B, N, d = entity_emb.shape
        K = head_idx.shape[1]

        # [B, K, d]
        h = entity_emb[torch.arange(B).unsqueeze(1), head_idx]  # [B, K, d]
        t = entity_emb[torch.arange(B).unsqueeze(1), tail_idx]  # [B, K, d]
        r = self.w_relation[rel_idx]  # [B, K, d//2]

        # RotatE
        h = h * self.embedding_range / math.sqrt(3.0)
        t = t * self.embedding_range / math.sqrt(3.0)
        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.embedding_range / math.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0).norm(dim=0).sum(dim=-1)  # [B, K]
        score = self.gamma - score
        return score

    def loss(self, entity_emb, pos_triples, neg_triples):
        """
        pos_triples [B, K, 3], neg_triples [B, K, neg_size, 3]
        """
        pos_head, pos_rel, pos_tail = pos_triples  # 解包出来的是 tensor
        B, K = pos_head.shape

        neg_head, neg_rel, neg_tail = neg_triples
        neg_size = neg_head.shape[2]

        # [B, K]
        pos_score = self.score(entity_emb, pos_head, pos_rel, pos_tail)

        neg_head = neg_head.reshape(B, -1)
        neg_rel = neg_rel.reshape(B, -1)
        neg_tail = neg_tail.reshape(B, -1)
        neg_score = self.score(entity_emb, neg_head, neg_rel, neg_tail).reshape(B, K, neg_size)


        neg_weight = F.softmax(neg_score * self.args.adversarial_temperature, dim=-1).detach()
        neg_loss = -(neg_weight * F.logsigmoid(-neg_score)).sum(dim=-1).mean()
        pos_loss = -F.logsigmoid(pos_score).mean()
        reg_loss = self.w_relation.pow(2).mean() * self.args.reg_param

        return (pos_loss + neg_loss) / 2 + reg_loss



def batch_tokenize_text(text_attributes, tokenizer, model, max_len=32, device='cuda'):
    """
   text_attributes [B, N] →  [B, N, d]
    """
    B = len(text_attributes)
    N = len(text_attributes[0]) if B > 0 else 0

    # flat_text = text_attributes.reshape(-1)  # [B*N]
    flat_text = [text for batch in text_attributes for text in batch]

    # 批量编码
    inputs = tokenizer(
        flat_text,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)

    with torch.no_grad():
        word_embeds = model.model.embed_tokens(inputs["input_ids"])  # [B*N, seq_len, d]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        text_emb = (word_embeds * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # [B*N, d]
        text_emb = text_emb.reshape(B, N, -1)  # [B, N, d]

    return text_emb


class GraphGrasper(nn.Module):
    def __init__(self, args, model, tokenizer):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.gat = GAT(
            args=args,
            in_channels=args.gnn_in_dim,
            hidden_channels=args.gnn_hidden_dim,
            out_channels=args.gnn_out_dim,
            n_ntype=args.n_ntype,
            n_etype=args.n_etype,
            heads=args.gnn_heads,
            gnn_layers=args.gnn_layers
        )

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_out_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            # nn.BatchNorm()???
            nn.Linear(1024, args.llm_emb_dim)
        ).to(self.device)

        self.link_decoder = RotatEDecoder(
            args=args,
            num_rels=args.n_etype,
            h_dim=args.llm_emb_dim
        ).to(self.device)

        self.concat_proj = nn.Linear(2 * args.llm_emb_dim, args.llm_emb_dim).to(self.device)

    def forward(self, gat_inputs, graphs, text_attributes, triples_list):
        """
        :param gat_inputs: [B*N, gnn_in_dim]
        :param graphs: list -> PyG Data
        :param text_attributes: [B,N] -> batch
        :param triples_list: list ->batch  [K,3]
        :return: multimodal_embedding [B, N, llm_emb_dim], link_loss(scalar_
        """
        B = len(graphs)
        N = graphs[0].num_nodes
        llm_dim = self.args.llm_emb_dim

        # [B*N, gnn_in_dim] -> [B*N, gnn_out_dim]
        batch_edge_index = []
        batch_edge_type = []
        batch_node_type = []
        offset = 0 # to distinguish different batch's entities
        for g in graphs:
            batch_edge_index.append(g.edge_index + offset)
            batch_edge_type.append(g.edge_type)
            batch_node_type.append(g.node_type)
            offset += g.num_nodes
        batch_edge_index = torch.cat(batch_edge_index, dim=1).to(self.device)
        batch_edge_type = torch.cat(batch_edge_type, dim=0).to(self.device)
        batch_node_type = torch.cat(batch_node_type, dim=0).to(self.device)

        # GAT forward
        graph_embedding = self.gat(
            args = self.args,
            x = gat_inputs.to(self.device),
            edge_index=batch_edge_index,
            edge_type=batch_edge_type,
            node_type=batch_node_type
        )

        graph_embedding_for_llm = self.projector(graph_embedding)
        text_embedding = batch_tokenize_text(
            text_attributes=text_attributes,
            tokenizer=self.tokenizer,
            model=self.model,
            max_len=self.args.max_text_len,
            device=self.device
        )

        graph_emb_reshape = graph_embedding_for_llm.reshape(B, N, llm_dim)
        concat_embedding = torch.cat(
            [graph_emb_reshape, text_embedding], dim=-1
        )  # [B, N, 2d]

        # [B, N, llm_dim]
        inputs_embeds = self.concat_proj(concat_embedding)  # [B, N, llm_dim]

        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )

        # outputs.last_hidden_state: [B, N, llm_dim]
        multimodal_emb = outputs.last_hidden_state  # [B, N, llm_dim]
        multimodal_embedding = multimodal_emb  # 已经是正确形状

        # link_loss flat
        multimodal_emb_flat = multimodal_emb.reshape(B * N, llm_dim)


        with torch.no_grad():
            pos_triples_batch = []
            neg_triples_batch = []
            for triples in triples_list:
                pos_tri, neg_tri = generate_pos_neg_triples(
                    triples=triples,
                    n_nodes=N,
                    neg_sample_size=self.args.neg_sample_size
                )
                pos_triples_batch.append(pos_tri)
                neg_triples_batch.append(neg_tri)

            # [B, K]（pos）、[B, K, neg_size]（neg）
            pos_triples = (
                torch.stack([pt[0] for pt in pos_triples_batch], dim=0),
                torch.stack([pt[1] for pt in pos_triples_batch], dim=0),
                torch.stack([pt[2] for pt in pos_triples_batch], dim=0)
            )
            neg_triples = (
                torch.stack([nt[0] for nt in neg_triples_batch], dim=0),
                torch.stack([nt[1] for nt in neg_triples_batch], dim=0),
                torch.stack([nt[2] for nt in neg_triples_batch], dim=0)
            )

        link_loss = self.link_decoder.loss(
            entity_emb=multimodal_embedding,
            pos_triples=pos_triples,
            neg_triples=neg_triples
        )

        return multimodal_embedding, link_loss


