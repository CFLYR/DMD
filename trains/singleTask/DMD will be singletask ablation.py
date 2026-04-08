import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss

logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def do_train(self, model, dataloader, return_epoch_results=False):
        """
        Training loop with conditional loss calculation based on ablation flags
        Supports best model checkpointing (saves only when validation metric improves)
        """
        # Get ablation flags from args
        use_FD = getattr(self.args, 'use_FD', True)
        use_HomoGD = getattr(self.args, 'use_HomoGD', True)
        use_CA = getattr(self.args, 'use_CA', True)
        use_HeteroGD = getattr(self.args, 'use_HeteroGD', True)
        
        # Get loss weights
        lambda_1 = getattr(self.args, 'lambda_1', 0.1)  # Decoupling loss weight
        lambda_2 = getattr(self.args, 'lambda_2', 0.05)  # Graph distillation loss weight
        gamma = getattr(self.args, 'gamma', 0.1)  # Orthogonality & margin weight

        # 0: DMD model, 1: Homo GD, 2: Hetero GD
        params = list(model[0].parameters())
        if use_HomoGD:
            params += list(model[1].parameters())
        if use_HeteroGD:
            params += list(model[2].parameters())

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        # Read max_epochs from args (can be dict-style or attribute-style due to edict)
        max_epochs = self.args.get('epochs', self.args.epochs if hasattr(self.args, 'epochs') else 30)
        max_epochs = int(max_epochs)  # Ensure it's an integer
        
        logger.info(f"DEBUG: Trainer initialized with max_epochs = {max_epochs}")
        print(f"\n{'='*80}")
        print(f"DEBUG: TRAINER do_train() INITIALIZED")
        print(f"  max_epochs resolved to: {max_epochs}")
        print(f"  self.args.get('epochs'): {self.args.get('epochs', 'NOT SET')}")
        print(f"  self.args.epochs (if exists): {self.args.epochs if hasattr(self.args, 'epochs') else 'ATTR NOT FOUND'}")
        print(f"{'='*80}\n")
        
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_dmd = model[0]
        net_distill_homo = model[1] if use_HomoGD else None
        net_distill_hetero = model[2] if use_HeteroGD else None
        net.append(net_dmd)
        if net_distill_homo:
            net.append(net_distill_homo)
        if net_distill_hetero:
            net.append(net_distill_hetero)
        model = net

        while epochs < max_epochs:
            epochs += 1
            
            # HARD SAFEGUARD: Explicitly check epoch limit
            if epochs > max_epochs:
                logger.info(f"DEBUG: HARD STOP - epoch {epochs} exceeds max_epochs {max_epochs}")
                print(f"\nDEBUG: HARD STOP AT EPOCH {epochs} (max_epochs={max_epochs})")
                break
            y_pred, y_true = [], []
            for mod in model:
                if mod is not None:
                    mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model[0](text, audio, vision, is_distill=True)

                    # TASK LOSS (always computed)
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task = loss_task_all
                    
                    # Add auxiliary task losses if outputs exist
                    if use_HomoGD and 'logits_l_homo' in output and output['logits_l_homo'] is not None:
                        loss_task += self.criterion(output['logits_l_homo'], labels)
                        loss_task += self.criterion(output['logits_v_homo'], labels)
                        loss_task += self.criterion(output['logits_a_homo'], labels)
                    if use_HomoGD and use_FD and 'logits_c' in output and output['logits_c'] is not None:
                        loss_task += self.criterion(output['logits_c'], labels)
                    if use_HeteroGD and use_CA and 'logits_l_hetero' in output and output['logits_l_hetero'] is not None:
                        loss_task += self.criterion(output['logits_l_hetero'], labels)
                        loss_task += self.criterion(output['logits_v_hetero'], labels)
                        loss_task += self.criterion(output['logits_a_hetero'], labels)

                    # DECOUPLING LOSS (only if use_FD=True)
                    loss_decoupling = 0.0
                    if use_FD:
                        # Reconstruction loss
                        if 'recon_l' in output and output['recon_l'] is not None:
                            loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                            loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                            loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                            loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                            # Cycle consistency loss
                            loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                            loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                            loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                            loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                            # Orthogonality loss
                            target_ort = torch.full((output['s_l_pooled'].size(0),), -1).to(self.args.device)
                            cosine_similarity_s_c_l = self.cosine(output['s_l_pooled'], output['c_l_pooled'], target_ort)
                            cosine_similarity_s_c_v = self.cosine(output['s_v_pooled'], output['c_v_pooled'], target_ort)
                            cosine_similarity_s_c_a = self.cosine(output['s_a_pooled'], output['c_a_pooled'], target_ort)
                            loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                            # Margin loss
                            c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                            ids, feats = [], []
                            for i in range(labels.size(0)):
                                feats.append(c_l[i].view(1, -1))
                                feats.append(c_v[i].view(1, -1))
                                feats.append(c_a[i].view(1, -1))
                                ids.append(labels[i].view(1, -1))
                                ids.append(labels[i].view(1, -1))
                                ids.append(labels[i].view(1, -1))
                            feats = torch.cat(feats, dim=0)
                            ids = torch.cat(ids, dim=0)
                            loss_sim = self.sim_loss(ids, feats)

                            # Combine decoupling losses with gamma weight
                            loss_decoupling = (loss_s_sr + loss_recon + (loss_sim + loss_ort) * gamma) * lambda_1

                    # GRAPH DISTILLATION LOSS (only if use_HomoGD or use_HeteroGD=True)
                    graph_distill_loss_homo = 0.0
                    graph_distill_loss_hetero = 0.0
                    
                    if use_HomoGD and 'logits_l_homo' in output and output['logits_l_homo'] is not None:
                        logits_homo = torch.stack([
                            output['logits_l_homo'],
                            output['logits_v_homo'],
                            output['logits_a_homo']
                        ])
                        reprs_homo = torch.stack([
                            output['repr_l_homo'],
                            output['repr_v_homo'],
                            output['repr_a_homo']
                        ])
                        
                        # edges for homo distill
                        edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)
                        loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                            model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                        graph_distill_loss_homo = lambda_2 * (loss_logit_homo + loss_reg_homo)

                    if use_HeteroGD and 'logits_l_hetero' in output and output['logits_l_hetero'] is not None:
                        logits_hetero = torch.stack([
                            output['logits_l_hetero'],
                            output['logits_v_hetero'],
                            output['logits_a_hetero']
                        ])
                        reprs_hetero = torch.stack([
                            output['repr_l_hetero'],
                            output['repr_v_hetero'],
                            output['repr_a_hetero']
                        ])
                        
                        # edges for hetero distill
                        edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)
                        loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
                            model[2].distillation_loss(logits_hetero, reprs_hetero, edges_hetero)
                        graph_distill_loss_hetero = lambda_2 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    # COMBINED LOSS
                    combined_loss = loss_task + loss_decoupling + graph_distill_loss_homo + graph_distill_loss_hetero

                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        params_to_clip = list(model[0].parameters())
                        if use_HomoGD:
                            params_to_clip += list(model[1].parameters())
                        if use_HeteroGD:
                            idx = 2 if use_HomoGD else 1
                            params_to_clip += list(model[idx].parameters())
                        nn.utils.clip_grad_value_(params_to_clip, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            
            # DEBUG: Log epoch progression with max_epochs check
            print(f"DEBUG: Epoch {epochs}/{max_epochs} (early_stop threshold: {self.args.early_stop}, best_epoch: {best_epoch})")
            
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            
            # CRITICAL: Best model checkpointing - save only when validation metric improves
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save best model
                model_save_path = self.args.get('model_save_path', './pt/dmd.pth')
                torch.save(model[0].state_dict(), model_save_path)
                logger.info(f">> Saved best model at epoch {epochs} with {self.args.KeyEval}={cur_valid:.4f}")

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            
            # Early stop OR max epochs reached
            if epochs - best_epoch >= self.args.early_stop:
                logger.info(f">> Early stopping at epoch {epochs}. Best epoch: {best_epoch}")
                print(f"DEBUG: EARLY STOP - No improvement for {self.args.early_stop} epochs")
                break
            if epochs >= max_epochs:
                logger.info(f">> Reached maximum epochs ({max_epochs}). Best epoch: {best_epoch} with {self.args.KeyEval}={best_valid:.4f}")
                print(f"DEBUG: MAX EPOCHS REACHED - Stopping at epoch {epochs}")
                break
        
        return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision, is_distill=True)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results