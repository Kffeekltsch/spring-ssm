import torch.optim as optim
def configure_optimizer_seq_layer(model, base_learning_rate, core_lr_factor=0.1, base_weight_decay=0.0, other_weight_decay=0.04):

    core_ssm_params = []
    other_params = []

    # Keywords
 
    ssm_core_param_names = [
        'Lambda',
        'B',
        'C',
        'log_step',
        'B_bias',
        'C_bias'
    ]

    print("--- Configuring SequenceLayer Optimizer Parameter Groups ---")
    assigned_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_core = False
        parts = name.split('.')
        if len(parts) >= 5: 
            if (parts[-1] in ssm_core_param_names and
                parts[-2] == 'seq' and
                parts[-3] == 's5'):
                 is_core = True 

        if is_core:
            core_ssm_params.append(param)
            assigned_params.add(name)
          

    # Assign remaining parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name not in assigned_params:
            other_params.append(param)


    # --- Verification ---
    total_core_params = sum(p.numel() for p in core_ssm_params)
    total_other_params = sum(p.numel() for p in other_params)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Identified Core SSM Params: {total_core_params}")
    print(f"Identified Other Params: {total_other_params}")
    print(f"Total Trainable Params: {total_trainable_params}")

    # --- Handle potential identification failures ---
    if total_core_params + total_other_params != total_trainable_params:
        print("ERROR: Parameter count mismatch! Not all trainable parameters assigned.")
        print("       Defaulting to single optimizer group.")
        param_groups = [{'params': list(model.parameters()), 'lr': base_learning_rate, 'weight_decay': other_weight_decay}]
    elif total_core_params == 0:

        print("         Applying 4x base LR and other WD to ALL parameters for now.")
        param_groups = [{'params': list(model.parameters()), 'lr': base_learning_rate * 4.0, 'weight_decay': other_weight_decay}]
    else:

        print(f"Applying LR={base_learning_rate * core_lr_factor}, WD={base_weight_decay} to Core SSM group.")
        print(f"Applying LR={base_learning_rate * 4.0}, WD={other_weight_decay} to Other group.")
        param_groups = [
            {
                'params': core_ssm_params,
                'lr': base_learning_rate * core_lr_factor, 
                'weight_decay': base_weight_decay          
            },
            {
                'params': other_params,
                'lr': base_learning_rate * 4.0,            
                'weight_decay': other_weight_decay         
            }
        ]

    #  AdamW optimizer
    optimizer = optim.AdamW(param_groups, lr=base_learning_rate)

    print("Optimizer configured.")
    return optimizer


def create_optimizer(model, config):
    """Factory function to create the correct optimizer."""
    lr = config['learning_rate']
    wd = config.get('weight_decay', 1e-4)

    if config.get('optimizer', {}).get('type') == 'differential':
        print("Configuring differential optimizer...")
        return configure_optimizer_seq_layer(
            model,
            lr,
            core_lr_factor=config['optimizer']['core_lr_factor'],
            other_weight_decay=config['optimizer']['other_weight_decay']
        )
    else:
        print("Configuring standard AdamW optimizer...")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)