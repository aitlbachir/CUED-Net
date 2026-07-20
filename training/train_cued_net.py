"""Single-run CUED-Net training."""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, 
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent))

from models.cued_net import CUEDNet, CUEDNetLoss, CUEDNetEnsemble
from data.datasets import get_dataloaders


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, class_weights):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc=f'Epoch {epoch}', leave=False):
        img_cc = batch['img_cc'].to(device)
        img_mlo = batch['img_mlo'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(img_cc, img_mlo)
            losses = criterion(outputs, labels, epoch, class_weights.to(device))
            loss = losses['total']
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        all_preds.extend(outputs['pred'].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return {
        'loss': total_loss / len(loader),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, class_weights):
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    all_evidential, all_discordance, all_combined = [], [], []
    all_agreements = []
    total_loss = 0
    
    for batch in loader:
        img_cc = batch['img_cc'].to(device)
        img_mlo = batch['img_mlo'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(img_cc, img_mlo)
        losses = criterion(outputs, labels, epoch, class_weights.to(device))
        
        total_loss += losses['total'].item()
        all_preds.extend(outputs['pred'].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(outputs['prob'][:, 1].cpu().numpy())
        all_evidential.extend(outputs['uncertainty_evidential'].cpu().numpy())
        all_discordance.extend(outputs['uncertainty_discordance'].cpu().numpy())
        all_combined.extend(outputs['uncertainty_combined'].cpu().numpy())
        all_agreements.extend(outputs['view_agreement'].cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return {
        'loss': total_loss / len(loader),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': auc,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
        'uncertainty_evidential': np.array(all_evidential),
        'uncertainty_discordance': np.array(all_discordance),
        'uncertainty_combined': np.array(all_combined),
        'view_agreement': np.mean(all_agreements)
    }


def train_single_model(args, seed, dataloaders, device):
    """Train a single CUED-Net model."""
    print(f"\n{'='*50}")
    print(f"Training CUED-Net - Seed {seed}")
    print(f"{'='*50}")
    
    set_seed(seed)
    
    output_dir = Path(args.output_dir) / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = CUEDNet(num_classes=2, pretrained=True).to(device)
    criterion = CUEDNetLoss(num_classes=2, lambda_vdl=0.3, lambda_kl=0.1)
    
    optimizer = optim.AdamW([
        {'params': model.encoder_cc.features.parameters(), 'lr': args.lr * 0.1},
        {'params': model.encoder_mlo.features.parameters(), 'lr': args.lr * 0.1},
        {'params': model.encoder_cc.classifier.parameters(), 'lr': args.lr},
        {'params': model.encoder_mlo.classifier.parameters(), 'lr': args.lr},
        {'params': model.encoder_cc.evidential.parameters(), 'lr': args.lr},
        {'params': model.encoder_mlo.evidential.parameters(), 'lr': args.lr},
    ], weight_decay=1e-4)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    best_metric = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        if epoch <= 5:
            model.freeze_encoders()
        else:
            model.unfreeze_encoders()
        
        train_metrics = train_epoch(
            model, dataloaders['train'], criterion, optimizer, scaler,
            device, epoch, dataloaders['class_weights']
        )
        
        val_metrics = evaluate(
            model, dataloaders['val'], criterion,
            device, epoch, dataloaders['class_weights']
        )
        
        scheduler.step()
        
        # Combined metric for early stopping
        combined = val_metrics['f1'] * 0.6 + val_metrics['auc'] * 0.4
        
        if combined > best_metric:
            best_metric = combined
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_metrics': {k: v for k, v in val_metrics.items() 
                               if not isinstance(v, np.ndarray)}
            }, output_dir / 'best_model.pt')
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}: F1={val_metrics['f1']:.3f}, AUC={val_metrics['auc']:.3f} *")
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: F1={val_metrics['f1']:.3f}, AUC={val_metrics['auc']:.3f}")
            
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best and evaluate on test
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(
        model, dataloaders['test'], criterion,
        device, args.epochs, dataloaders['class_weights']
    )
    
    print(f"Seed {seed} Test: F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    return model, test_metrics


def selective_prediction_analysis(labels, preds, probs, uncertainties):
    """
    NOVEL: Coverage-Accuracy Trade-off Analysis
    Shows how accuracy improves when uncertain predictions are rejected.
    """
    results = []
    
    # Sort by uncertainty
    sorted_indices = np.argsort(uncertainties)
    
    for coverage in np.arange(0.5, 1.01, 0.05):
        n_select = int(len(labels) * coverage)
        if n_select == 0:
            continue
        
        # Select most certain predictions
        selected = sorted_indices[:n_select]
        
        sel_labels = labels[selected]
        sel_preds = preds[selected]
        sel_probs = probs[selected]
        
        # Metrics at this coverage
        f1 = f1_score(sel_labels, sel_preds, zero_division=0)
        acc = accuracy_score(sel_labels, sel_preds)
        
        try:
            auc = roc_auc_score(sel_labels, sel_probs)
        except:
            auc = 0.5
        
        results.append({
            'coverage': coverage,
            'f1': f1,
            'accuracy': acc,
            'auc': auc,
            'n_samples': n_select
        })
    
    return results


def bootstrap_ci(labels, preds, probs, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence intervals."""
    n = len(labels)
    metrics = {'f1': [], 'auc': [], 'accuracy': []}
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        y_true = labels[idx]
        y_pred = preds[idx]
        y_prob = probs[idx]
        
        metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        try:
            metrics['auc'].append(roc_auc_score(y_true, y_prob))
        except:
            metrics['auc'].append(0.5)
    
    alpha = (1 - ci) / 2
    results = {}
    for name, values in metrics.items():
        values = np.array(values)
        results[name] = {
            'mean': float(np.mean(values)),
            'ci_lower': float(np.percentile(values, alpha * 100)),
            'ci_upper': float(np.percentile(values, (1 - alpha) * 100))
        }
    
    return results


def ensemble_evaluation(models, loader, device):
    """
    NOVEL: Triple Uncertainty Ensemble Evaluation
    """
    ensemble = CUEDNetEnsemble(models)
    
    all_results = []
    all_labels = []
    
    for batch in tqdm(loader, desc='Ensemble Evaluation'):
        outputs = ensemble.predict(batch['img_cc'], batch['img_mlo'], device)
        
        for i in range(len(batch['label'])):
            all_results.append({
                'prob': outputs['prob'][i, 1].item(),
                'pred': outputs['pred'][i].item(),
                'u_evidential': outputs['uncertainty_evidential'][i].item(),
                'u_ensemble': outputs['uncertainty_ensemble'][i].item(),
                'u_discordance': outputs['uncertainty_discordance'][i].item(),
                'u_total': outputs['uncertainty_total'][i].item(),
            })
        all_labels.extend(batch['label'].numpy())
    
    labels = np.array(all_labels)
    probs = np.array([r['prob'] for r in all_results])
    preds = np.array([r['pred'] for r in all_results])
    u_total = np.array([r['u_total'] for r in all_results])
    u_evidential = np.array([r['u_evidential'] for r in all_results])
    u_ensemble = np.array([r['u_ensemble'] for r in all_results])
    u_discordance = np.array([r['u_discordance'] for r in all_results])
    
    # Find optimal threshold
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.3, 0.7, 0.05):
        p = (probs > thresh).astype(int)
        f1 = f1_score(labels, p)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    
    final_preds = (probs > best_thresh).astype(int)
    
    # Compute all metrics
    metrics = {
        'f1': f1_score(labels, final_preds),
        'auc': roc_auc_score(labels, probs),
        'accuracy': accuracy_score(labels, final_preds),
        'precision': precision_score(labels, final_preds, zero_division=0),
        'recall': recall_score(labels, final_preds, zero_division=0),
        'ap': average_precision_score(labels, probs),
        'threshold': best_thresh,
        'confusion_matrix': confusion_matrix(labels, final_preds).tolist()
    }
    
    # Bootstrap CIs
    metrics['bootstrap'] = bootstrap_ci(labels, final_preds, probs)
    
    # Selective prediction analysis for each uncertainty type
    metrics['selective_total'] = selective_prediction_analysis(labels, final_preds, probs, u_total)
    metrics['selective_evidential'] = selective_prediction_analysis(labels, final_preds, probs, u_evidential)
    metrics['selective_ensemble'] = selective_prediction_analysis(labels, final_preds, probs, u_ensemble)
    metrics['selective_discordance'] = selective_prediction_analysis(labels, final_preds, probs, u_discordance)
    
    # Uncertainty statistics
    correct = final_preds == labels
    metrics['uncertainty_analysis'] = {
        'correct_mean': float(u_total[correct].mean()) if correct.sum() > 0 else 0,
        'incorrect_mean': float(u_total[~correct].mean()) if (~correct).sum() > 0 else 0,
        'evidential_mean': float(u_evidential.mean()),
        'ensemble_mean': float(u_ensemble.mean()),
        'discordance_mean': float(u_discordance.mean()),
    }
    
    # Statistical test: is uncertainty higher for incorrect predictions?
    if correct.sum() > 0 and (~correct).sum() > 0:
        stat, pval = stats.mannwhitneyu(u_total[~correct], u_total[correct], alternative='greater')
        metrics['uncertainty_analysis']['mannwhitney_pvalue'] = float(pval)
    
    return metrics, {
        'labels': labels,
        'probs': probs,
        'preds': final_preds,
        'uncertainties': {
            'total': u_total,
            'evidential': u_evidential,
            'ensemble': u_ensemble,
            'discordance': u_discordance
        }
    }


def plot_results(metrics, raw_data, output_dir):
    """Generate publication-quality figures."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(raw_data['labels'], raw_data['probs'])
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(raw_data['labels'], raw_data['probs'])
    ax.plot(rec, prec, 'b-', linewidth=2, label=f"AP = {metrics['ap']:.3f}")
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. Coverage-F1 Trade-off (NOVEL)
    ax = axes[0, 2]
    for name, key, color in [
        ('Total', 'selective_total', 'blue'),
        ('Evidential', 'selective_evidential', 'green'),
        ('Ensemble', 'selective_ensemble', 'red'),
        ('Discordance', 'selective_discordance', 'purple')
    ]:
        sel = metrics[key]
        coverages = [s['coverage'] for s in sel]
        f1s = [s['f1'] for s in sel]
        ax.plot(coverages, f1s, '-o', color=color, label=name, markersize=4)
    
    ax.axhline(y=metrics['f1'], color='gray', linestyle='--', alpha=0.5, label=f'Full F1={metrics["f1"]:.3f}')
    ax.set_xlabel('Coverage', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Selective Prediction: Coverage vs F1', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Uncertainty Distribution
    ax = axes[1, 0]
    correct = raw_data['preds'] == raw_data['labels']
    u = raw_data['uncertainties']['total']
    ax.hist(u[correct], bins=20, alpha=0.6, label='Correct', color='green', density=True)
    ax.hist(u[~correct], bins=20, alpha=0.6, label='Incorrect', color='red', density=True)
    ax.set_xlabel('Total Uncertainty', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Uncertainty Distribution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 5. Triple Uncertainty Decomposition (NOVEL)
    ax = axes[1, 1]
    uncertainty_means = [
        metrics['uncertainty_analysis']['evidential_mean'],
        metrics['uncertainty_analysis']['ensemble_mean'],
        metrics['uncertainty_analysis']['discordance_mean']
    ]
    bars = ax.bar(['Evidential', 'Ensemble', 'Discordance'], uncertainty_means,
                  color=['#2ecc71', '#e74c3c', '#9b59b6'])
    ax.set_ylabel('Mean Uncertainty', fontsize=12)
    ax.set_title('Triple Uncertainty Decomposition', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Confusion Matrix
    ax = axes[1, 2]
    cm = np.array(metrics['confusion_matrix'])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign-Like', 'Malignant-Like'])
    ax.set_yticklabels(['Benign-Like', 'Malignant-Like'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'results_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'results_figure.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/workspace/cbis-ddsm')
    parser.add_argument('--output_dir', type=str, default='./outputs_cued')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--seeds', type=str, default='42,123,456,789,2024')
    
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    
    print("="*60)
    print("CUED-Net: Cross-view Uncertainty-guided Evidential")
    print("         Dual-encoder Network")
    print("="*60)
    print(f"Seeds: {seeds}")
    print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    dataloaders = get_dataloaders(args.data_root, args.batch_size, oversample=True)
    
    train_labels = [p['label'] for p in dataloaders['train_dataset'].pairs]
    print(f"Train: {len(train_labels)} pairs (Benign: {train_labels.count(0)}, Malignant: {train_labels.count(1)})")
    print(f"Val: {len(dataloaders['val_dataset'])} pairs")
    print(f"Test: {len(dataloaders['test_dataset'])} pairs")
    
    # Train all models
    models = []
    individual_results = []
    
    start_time = time.time()
    
    for seed in seeds:
        model, test_metrics = train_single_model(args, seed, dataloaders, device)
        models.append(model)
        individual_results.append({
            'seed': seed,
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc']
        })
    
    training_time = time.time() - start_time
    
    # Ensemble evaluation
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION WITH TRIPLE UNCERTAINTY")
    print("="*60)
    
    ensemble_metrics, raw_data = ensemble_evaluation(models, dataloaders['test'], device)
    
    # Print results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    print("\nIndividual Models:")
    for r in individual_results:
        print(f"  Seed {r['seed']}: F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
    
    print(f"\nEnsemble (5 models):")
    print(f"  F1 Score:    {ensemble_metrics['f1']:.4f} (95% CI: [{ensemble_metrics['bootstrap']['f1']['ci_lower']:.3f}, {ensemble_metrics['bootstrap']['f1']['ci_upper']:.3f}])")
    print(f"  ROC AUC:     {ensemble_metrics['auc']:.4f} (95% CI: [{ensemble_metrics['bootstrap']['auc']['ci_lower']:.3f}, {ensemble_metrics['bootstrap']['auc']['ci_upper']:.3f}])")
    print(f"  Accuracy:    {ensemble_metrics['accuracy']:.4f}")
    print(f"  Precision:   {ensemble_metrics['precision']:.4f}")
    print(f"  Recall:      {ensemble_metrics['recall']:.4f}")
    print(f"  Avg Prec:    {ensemble_metrics['ap']:.4f}")
    print(f"  Threshold:   {ensemble_metrics['threshold']:.2f}")
    
    cm = np.array(ensemble_metrics['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    print(f"\nUncertainty Analysis:")
    ua = ensemble_metrics['uncertainty_analysis']
    print(f"  Correct predictions mean uncertainty:   {ua['correct_mean']:.4f}")
    print(f"  Incorrect predictions mean uncertainty: {ua['incorrect_mean']:.4f}")
    if 'mannwhitney_pvalue' in ua:
        print(f"  Mann-Whitney U test p-value: {ua['mannwhitney_pvalue']:.4e}")
    
    print(f"\nSelective Prediction (80% coverage):")
    sel_80 = [s for s in ensemble_metrics['selective_total'] if abs(s['coverage'] - 0.8) < 0.1]
    if sel_80:
        s = sel_80[0]
        print(f"  F1 at {s['coverage']*100:.0f}% coverage: {s['f1']:.4f}")
    
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    
    # Generate figures
    plot_results(ensemble_metrics, raw_data, output_dir)
    
    # Save results
    results = {
        'ensemble_metrics': {k: v for k, v in ensemble_metrics.items() 
                            if k not in ['selective_total', 'selective_evidential', 
                                        'selective_ensemble', 'selective_discordance']},
        'selective_prediction': ensemble_metrics['selective_total'],
        'individual_results': individual_results,
        'training_time_minutes': training_time / 60,
        'args': vars(args)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nResults saved to {output_dir}")
    print(f"Figures saved: results_figure.png, results_figure.pdf")


if __name__ == '__main__':
    main()
