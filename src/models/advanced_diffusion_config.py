# 高级训练配置
ADVANCED_CONFIG = {
    # 使用更大的模型
    'hidden_dim': 512,
    'num_layers': 6,
    
    # 更精细的噪声调度
    'num_timesteps': 2000,
    'beta_schedule': 'cosine',  # 或 'linear'
    
    # 混合精度训练
    'use_fp16': True,
    
    # 数据增强
    'use_augmentation': True,
    
    # 早停机制
    'early_stopping': True,
    'patience': 20,
    
    # 学习率调度
    'lr_scheduler': 'cosine',
    'warmup_steps': 1000
} 