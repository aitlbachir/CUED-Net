"""CUED-Net: dual-encoder evidential network with view-discordance uncertainty."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class EvidentialLayer(nn.Module):
    """
    Evidential output layer that produces Dirichlet parameters.
    Outputs evidence -> alpha -> probability + uncertainty
    """
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # Evidence must be non-negative
        evidence = F.softplus(self.fc(x))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        uncertainty = self.num_classes / S.squeeze(1)
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'prob': prob,
            'uncertainty': uncertainty,
            'strength': S.squeeze(1)
        }


class ViewEncoder(nn.Module):
    """Single view encoder with evidential output."""
    def __init__(self, num_classes=2, pretrained=True, hidden_dim=256):
        super().__init__()
        
        # DenseNet121 backbone
        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = densenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Evidential output
        self.evidential = EvidentialLayer(64, num_classes)
    
    def forward(self, x):
        features = self.features(x)
        features = self.pool(features).flatten(1)
        hidden = self.classifier(features)
        output = self.evidential(hidden)
        output['features'] = features
        return output


class CUEDNet(nn.Module):
    """
    CUED-Net: Cross-view Uncertainty-guided Evidential Dual-encoder Network
    
    Novel architecture that:
    1. Processes CC and MLO views independently
    2. Computes view discordance as additional uncertainty signal
    3. Fuses predictions with uncertainty-aware weighting
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Independent view encoders
        self.encoder_cc = ViewEncoder(num_classes, pretrained)
        self.encoder_mlo = ViewEncoder(num_classes, pretrained)
    
    def forward(self, img_cc, img_mlo):
        # Get view-specific predictions
        cc_out = self.encoder_cc(img_cc)
        mlo_out = self.encoder_mlo(img_mlo)
        
        # === NOVEL: View Discordance Uncertainty ===
        # Measures disagreement between views
        prob_cc = cc_out['prob']
        prob_mlo = mlo_out['prob']
        
        # KL divergence between view predictions (symmetrized)
        kl_cc_mlo = F.kl_div(prob_cc.log(), prob_mlo, reduction='none').sum(dim=1)
        kl_mlo_cc = F.kl_div(prob_mlo.log(), prob_cc, reduction='none').sum(dim=1)
        view_discordance = (kl_cc_mlo + kl_mlo_cc) / 2
        
        # Normalized view discordance (0 to 1)
        view_discordance_norm = torch.sigmoid(view_discordance)
        
        # === Uncertainty-Weighted Fusion ===
        # Weight each view by inverse of its uncertainty
        cc_confidence = 1.0 - cc_out['uncertainty']
        mlo_confidence = 1.0 - mlo_out['uncertainty']
        
        # Normalize weights
        total_conf = cc_confidence + mlo_confidence + 1e-8
        w_cc = cc_confidence / total_conf
        w_mlo = mlo_confidence / total_conf
        
        # Fused probability
        fused_prob = w_cc.unsqueeze(1) * prob_cc + w_mlo.unsqueeze(1) * prob_mlo
        
        # === Triple Uncertainty ===
        # 1. Evidential: average of view uncertainties
        evidential_uncertainty = (cc_out['uncertainty'] + mlo_out['uncertainty']) / 2
        
        # 2. View discordance: already computed
        # 3. Combined uncertainty
        combined_uncertainty = evidential_uncertainty + 0.5 * view_discordance_norm
        
        # Predictions
        pred = torch.argmax(fused_prob, dim=1)
        
        # View agreement
        cc_pred = torch.argmax(prob_cc, dim=1)
        mlo_pred = torch.argmax(prob_mlo, dim=1)
        view_agreement = (cc_pred == mlo_pred).float()
        
        return {
            'prob': fused_prob,
            'pred': pred,
            'cc_out': cc_out,
            'mlo_out': mlo_out,
            'uncertainty_evidential': evidential_uncertainty,
            'uncertainty_discordance': view_discordance_norm,
            'uncertainty_combined': combined_uncertainty,
            'view_agreement': view_agreement,
            'fusion_weights': {'cc': w_cc, 'mlo': w_mlo}
        }
    
    def freeze_encoders(self):
        for param in self.encoder_cc.features.parameters():
            param.requires_grad = False
        for param in self.encoder_mlo.features.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        for param in self.encoder_cc.features.parameters():
            param.requires_grad = True
        for param in self.encoder_mlo.features.parameters():
            param.requires_grad = True


class CUEDNetLoss(nn.Module):
    """
    Novel loss function for CUED-Net combining:
    1. Evidential loss (Type II MLE)
    2. View Discordance Loss (VDL) - penalizes confident contradictions
    3. Consistency regularization
    """
    def __init__(self, num_classes=2, lambda_vdl=0.3, lambda_kl=0.1, annealing_epochs=10):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_vdl = lambda_vdl
        self.lambda_kl = lambda_kl
        self.annealing_epochs = annealing_epochs
    
    def forward(self, outputs, targets, epoch=0, class_weights=None):
        cc_out = outputs['cc_out']
        mlo_out = outputs['mlo_out']
        
        # === Evidential Loss for both views ===
        loss_cc = self._evidential_loss(cc_out['alpha'], targets, epoch, class_weights)
        loss_mlo = self._evidential_loss(mlo_out['alpha'], targets, epoch, class_weights)
        evidential_loss = loss_cc + loss_mlo
        
        # === NOVEL: View Discordance Loss (VDL) ===
        # Penalize when views make confident but contradictory predictions
        cc_pred = torch.argmax(cc_out['prob'], dim=1)
        mlo_pred = torch.argmax(mlo_out['prob'], dim=1)
        
        # Views disagree
        disagreement = (cc_pred != mlo_pred).float()
        
        # Confidence of each view (1 - uncertainty)
        cc_conf = 1.0 - cc_out['uncertainty']
        mlo_conf = 1.0 - mlo_out['uncertainty']
        
        # VDL: penalize high confidence when views disagree
        # If views disagree but both are confident, loss is high
        vdl = disagreement * cc_conf * mlo_conf
        vdl_loss = vdl.mean()
        
        # === Consistency Regularization ===
        # Encourage similar predictions when both views see the same lesion
        consistency_loss = F.mse_loss(cc_out['prob'], mlo_out['prob'])
        
        # Anneal VDL and consistency
        anneal = min(1.0, epoch / self.annealing_epochs)
        
        total_loss = evidential_loss + \
                     self.lambda_vdl * anneal * vdl_loss + \
                     0.1 * anneal * consistency_loss
        
        return {
            'total': total_loss,
            'evidential': evidential_loss,
            'vdl': vdl_loss,
            'consistency': consistency_loss
        }
    
    def _evidential_loss(self, alpha, targets, epoch, class_weights):
        """Type II Maximum Likelihood with KL regularization."""
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        
        # One-hot targets
        y_onehot = F.one_hot(targets, self.num_classes).float()
        
        # MSE loss
        mse = torch.sum((y_onehot - prob) ** 2, dim=1)
        var = torch.sum(prob * (1 - prob) / (S + 1), dim=1)
        nll = mse + var
        
        # Class weights
        if class_weights is not None:
            nll = nll * class_weights[targets]
        
        # KL regularization
        alpha_tilde = y_onehot + (1 - y_onehot) * (alpha - 1)
        alpha_tilde = torch.clamp(alpha_tilde, min=1.0)
        kl = self._kl_divergence(alpha_tilde)
        
        anneal = min(1.0, epoch / self.annealing_epochs)
        
        return (nll + self.lambda_kl * anneal * kl).mean()
    
    def _kl_divergence(self, alpha):
        K = self.num_classes
        alpha0 = torch.sum(alpha, dim=1, keepdim=True)
        log_beta = torch.sum(torch.lgamma(alpha), dim=1) - torch.lgamma(alpha0.squeeze(1))
        log_beta_uniform = K * torch.lgamma(torch.tensor(1.0, device=alpha.device)) - \
                          torch.lgamma(torch.tensor(float(K), device=alpha.device))
        digamma_term = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha0)), dim=1)
        return log_beta - log_beta_uniform + digamma_term


class CUEDNetEnsemble:
    """
    Ensemble of CUED-Net models with triple uncertainty aggregation.
    
    Novel: Combines three types of uncertainty for robust prediction:
    1. Evidential (within-model)
    2. Ensemble (between-model variance)
    3. View Discordance (between-view)
    """
    def __init__(self, models):
        self.models = models
        self.num_models = len(models)
    
    @torch.no_grad()
    def predict(self, img_cc, img_mlo, device):
        all_probs = []
        all_evidential = []
        all_discordance = []
        all_agreements = []
        
        for model in self.models:
            model.eval()
            outputs = model(img_cc.to(device), img_mlo.to(device))
            
            all_probs.append(outputs['prob'].cpu())
            all_evidential.append(outputs['uncertainty_evidential'].cpu())
            all_discordance.append(outputs['uncertainty_discordance'].cpu())
            all_agreements.append(outputs['view_agreement'].cpu())
        
        # Stack
        probs = torch.stack(all_probs)  # (n_models, batch, classes)
        evidential = torch.stack(all_evidential)  # (n_models, batch)
        discordance = torch.stack(all_discordance)
        
        # === Triple Uncertainty Aggregation ===
        
        # 1. Mean prediction
        mean_prob = probs.mean(dim=0)
        
        # 2. Evidential uncertainty (average across models)
        mean_evidential = evidential.mean(dim=0)
        
        # 3. Ensemble uncertainty (variance of predictions)
        ensemble_var = probs[:, :, 1].var(dim=0)  # Variance of P(malignant)
        
        # 4. View discordance (average)
        mean_discordance = discordance.mean(dim=0)
        
        # 5. Combined uncertainty (weighted sum)
        total_uncertainty = (
            0.4 * mean_evidential + 
            0.3 * ensemble_var + 
            0.3 * mean_discordance
        )
        
        # Predictions
        pred = torch.argmax(mean_prob, dim=1)
        
        return {
            'prob': mean_prob,
            'pred': pred,
            'uncertainty_evidential': mean_evidential,
            'uncertainty_ensemble': ensemble_var,
            'uncertainty_discordance': mean_discordance,
            'uncertainty_total': total_uncertainty,
            'view_agreement': torch.stack(all_agreements).mean(dim=0)
        }
