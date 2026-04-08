"""
DMD backbone with ablation study support for feature decoupling and multimodal transformers
Supports 6 ablation variants via conditional module initialization and forward pass
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder

class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
        
        # Ablation flags - control which modules are active
        self.use_FD = getattr(args, 'use_FD', True)
        self.use_HomoGD = getattr(args, 'use_HomoGD', True)
        self.use_CA = getattr(args, 'use_CA', True)
        self.use_HeteroGD = getattr(args, 'use_HeteroGD', True)
        
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a
        
        # CRITICAL: Dynamic classifier dimension based on ablation flags
        if not self.use_FD:  # Variant 6: Baseline
            combined_dim = self.d_l + self.d_a + self.d_v
        elif self.use_FD and not self.use_HomoGD:  # Variant 5: Only FD
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        elif self.use_HomoGD and not self.use_CA and not self.use_HeteroGD:  # Variant 4: Only HomoGD
            combined_dim = self.d_l + self.d_a + self.d_v
        elif self.use_FD and self.use_HomoGD and not self.use_CA:  # Variant 3: No CA (with or without HeteroGD)
            if self.use_HeteroGD:
                combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
            else:
                combined_dim = self.d_l + self.d_a + self.d_v
        elif self.use_CA and not self.use_HeteroGD:  # Variant 2: No HeteroGD
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        else:  # Variant 1: Full model
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        
        output_dim = 1

        # 1. Temporal convolutional layers for initial feature (always needed)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Feature Decoupling modules (only if use_FD=True)
        if self.use_FD:
            # 2.1 Modality-specific encoder
            self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
            self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
            self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

            # 2.2 Modality-invariant encoder
            self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

            # 3. Decoder for reconstruct three modalities
            self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
            self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
            self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

            # for calculate cosine sim between s_x
            self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
            self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
            self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

            # for align c_l, c_v, c_a
            self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
            self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
            self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

            self.self_attentions_c_l = self.get_network(self_type='l')
            self.self_attentions_c_v = self.get_network(self_type='v')
            self.self_attentions_c_a = self.get_network(self_type='a')

            self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
            self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
            self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        # 4. Cross-modal Attention modules (only if use_CA=True)
        if self.use_CA:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
            self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
            self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
            self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 5. Homogeneous Graph Distillation modules (only if use_HomoGD=True)
        if self.use_HomoGD:
            self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
            self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
            self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
            self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
            self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
            self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
            self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
            self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
            self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        # 6. Heterogeneous Graph Distillation modules (only if use_HeteroGD=True)
        if self.use_HeteroGD:
            self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
            self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
            self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        # 7. Ensemble Projection layers (conditional)
        if self.use_CA and self.use_HeteroGD:
            # weight for each modality
            self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
            self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)
            self.weight_a = nn.Linear(2 * self.d_a, 2 * self.d_a)
            self.weight_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        
        # 8. Final classifier with DYNAMIC input dimension
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, is_distill=False):
        """
        Conditional forward pass based on ablation flags
        Supports 6 variants with different computational paths
        """
        # Initial processing (common to all variants)
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        # Initialize result dictionary with common outputs
        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
        }
        
        # VARIANT 6: Baseline - Direct concatenation (no FD, no GD, no CA)
        if not self.use_FD:
            # Direct concatenation after conv1d projection
            feat_l = proj_x_l.transpose(1, 2).contiguous().view(proj_x_l.size(0), -1)
            feat_a = proj_x_a.transpose(1, 2).contiguous().view(proj_x_a.size(0), -1)
            feat_v = proj_x_v.transpose(1, 2).contiguous().view(proj_x_v.size(0), -1)
            
            # Concatenate and classify
            last_hs = torch.cat([feat_l, feat_a, feat_v], dim=1)
            last_hs_proj = self.proj2(
                F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)
            
            res['output_logit'] = output
            # Return minimal outputs for variant 6
            return res
        
        # FEATURE DECOUPLING (Variants 1-5)
        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        c_list = [c_l, c_v, c_a]

        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))

        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        s_l_r = self.encoder_s_l(recon_l)
        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)

        # Pool features for orthogonality loss (before permute)
        s_l_pooled = s_l.mean(dim=2)
        s_v_pooled = s_v.mean(dim=2)
        s_a_pooled = s_a.mean(dim=2)
        c_l_pooled = c_l.mean(dim=2)
        c_v_pooled = c_v.mean(dim=2)
        c_a_pooled = c_a.mean(dim=2)

        s_l = s_l.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        proj_s_l = self.proj_cosine_l(s_l.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_v = self.proj_cosine_v(s_v.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_a = self.proj_cosine_a(s_a.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        
        # Add FD outputs to result
        res.update({
            's_l': s_l, 's_v': s_v, 's_a': s_a,
            's_l_pooled': s_l_pooled, 's_v_pooled': s_v_pooled, 's_a_pooled': s_a_pooled,
            'proj_s_l': proj_s_l, 'proj_s_v': proj_s_v, 'proj_s_a': proj_s_a,
            'c_l': c_l, 'c_v': c_v, 'c_a': c_a,
            'c_l_pooled': c_l_pooled, 'c_v_pooled': c_v_pooled, 'c_a_pooled': c_a_pooled,
            'c_l_sim': c_l_sim, 'c_v_sim': c_v_sim, 'c_a_sim': c_a_sim,
            's_l_r': s_l_r, 's_v_r': s_v_r, 's_a_r': s_a_r,
            'recon_l': recon_l, 'recon_v': recon_v, 'recon_a': recon_a,
        })
        
        # HOMOGENEOUS GD (Variants 1-4)
        if self.use_HomoGD:
            c_l_att = self.self_attentions_c_l(c_l)
            if type(c_l_att) == tuple:
                c_l_att = c_l_att[0]
            c_l_att = c_l_att[-1]
            c_v_att = self.self_attentions_c_v(c_v)
            if type(c_v_att) == tuple:
                c_v_att = c_v_att[0]
            c_v_att = c_v_att[-1]
            c_a_att = self.self_attentions_c_a(c_a)
            if type(c_a_att) == tuple:
                c_a_att = c_a_att[0]
            c_a_att = c_a_att[-1]
            c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)

            c_proj = self.proj2_c(
                F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                          training=self.training))
            c_proj += c_fusion
            logits_c = self.out_layer_c(c_proj)

            hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
            repr_l_low = self.proj1_l_low(hs_l_low)
            hs_proj_l_low = self.proj2_l_low(
                F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
            hs_proj_l_low += hs_l_low
            logits_l_low = self.out_layer_l_low(hs_proj_l_low)

            hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
            repr_v_low = self.proj1_v_low(hs_v_low)
            hs_proj_v_low = self.proj2_v_low(
                F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
            hs_proj_v_low += hs_v_low
            logits_v_low = self.out_layer_v_low(hs_proj_v_low)

            hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
            repr_a_low = self.proj1_a_low(hs_a_low)
            hs_proj_a_low = self.proj2_a_low(
                F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
            hs_proj_a_low += hs_a_low
            logits_a_low = self.out_layer_a_low(hs_proj_a_low)
            
            res.update({
                'logits_l_homo': logits_l_low,
                'logits_v_homo': logits_v_low,
                'logits_a_homo': logits_a_low,
                'repr_l_homo': repr_l_low,
                'repr_v_homo': repr_v_low,
                'repr_a_homo': repr_a_low,
                'logits_c': logits_c,
            })
        
        # CROSS-MODAL ATTENTION (Variants 1-2)
        if self.use_CA:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)
            h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls[-1]

            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(s_a, s_l, s_l)
            h_a_with_vs = self.trans_a_with_v(s_a, s_v, s_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as[-1]

            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(s_v, s_l, s_l)
            h_v_with_as = self.trans_v_with_a(s_v, s_a, s_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = h_vs[-1]
            
            res.update({
                'last_h_l': h_ls[-1],
                'last_h_v': h_vs[-1],
                'last_h_a': h_as[-1],
            })
        
        # HETEROGENEOUS GD (Variants 1, 3)
        if self.use_HeteroGD:
            if self.use_CA:
                # Use CA outputs for HeteroGD
                hs_proj_l_high = self.proj2_l_high(
                    F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
                hs_proj_l_high += last_h_l
                logits_l_high = self.out_layer_l_high(hs_proj_l_high)

                hs_proj_v_high = self.proj2_v_high(
                    F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
                hs_proj_v_high += last_h_v
                logits_v_high = self.out_layer_v_high(hs_proj_v_high)

                hs_proj_a_high = self.proj2_a_high(
                    F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                              training=self.training))
                hs_proj_a_high += last_h_a
                logits_a_high = self.out_layer_a_high(hs_proj_a_high)
            else:
                # Variant 3: HeteroGD without CA - use decoupled features directly
                # Concatenate s_x features for hetero processing
                s_l_flat = s_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
                s_v_flat = s_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
                s_a_flat = s_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
                
                # Simple projection (dimensions may need adjustment)
                # For Variant 3, we use the decoupled specific features
                hs_proj_l_high = s_l_flat
                hs_proj_v_high = s_v_flat
                hs_proj_a_high = s_a_flat
                logits_l_high = None  # Not applicable without CA
                logits_v_high = None
                logits_a_high = None
            
            res.update({
                'logits_l_hetero': logits_l_high,
                'logits_v_hetero': logits_v_high,
                'logits_a_hetero': logits_a_high,
                'repr_l_hetero': hs_proj_l_high,
                'repr_v_hetero': hs_proj_v_high,
                'repr_a_hetero': hs_proj_a_high,
            })
        
        # FINAL FUSION BASED ON VARIANT
        if not self.use_FD:
            # Already handled above (Variant 6)
            pass
        elif self.use_FD and not self.use_HomoGD:
            # Variant 5: Only FD - concatenate decoupled features
            s_l_flat = s_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
            s_v_flat = s_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
            s_a_flat = s_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
            c_l_flat = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
            c_v_flat = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
            c_a_flat = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
            
            last_hs = torch.cat([s_l_flat, s_v_flat, s_a_flat, c_l_flat, c_v_flat, c_a_flat], dim=1)
        elif self.use_HomoGD and not self.use_CA and not self.use_HeteroGD:
            # Variant 4: Only HomoGD - use homogeneous features
            last_hs = torch.cat([repr_l_low, repr_v_low, repr_a_low], dim=1)
        elif not self.use_CA and self.use_HeteroGD:
            # Variant 3: No CA but with HeteroGD
            s_l_flat = s_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
            s_v_flat = s_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
            s_a_flat = s_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
            c_l_flat = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
            c_v_flat = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
            c_a_flat = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
            
            last_hs = torch.cat([s_l_flat, s_v_flat, s_a_flat, c_l_flat, c_v_flat, c_a_flat], dim=1)
        elif self.use_CA and not self.use_HeteroGD:
            # Variant 2: No HeteroGD - use CA outputs + HomoGD
            last_h_l_weighted = torch.sigmoid(self.weight_l(last_h_l))
            last_h_v_weighted = torch.sigmoid(self.weight_v(last_h_v))
            last_h_a_weighted = torch.sigmoid(self.weight_a(last_h_a))
            c_fusion_weighted = torch.sigmoid(self.weight_c(c_fusion))
            
            last_hs = torch.cat([last_h_l_weighted, last_h_v_weighted, last_h_a_weighted, c_fusion_weighted], dim=1)
        else:
            # Variant 1: Full model - all components
            last_h_l_weighted = torch.sigmoid(self.weight_l(last_h_l))
            last_h_v_weighted = torch.sigmoid(self.weight_v(last_h_v))
            last_h_a_weighted = torch.sigmoid(self.weight_a(last_h_a))
            c_fusion_weighted = torch.sigmoid(self.weight_c(c_fusion))
            
            last_hs = torch.cat([last_h_l_weighted, last_h_v_weighted, last_h_a_weighted, c_fusion_weighted], dim=1)
        
        # Final projection and classification
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        
        res['output_logit'] = output
        return res