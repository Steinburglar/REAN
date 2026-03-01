experiments/
├── production/                                                     # Production experiment used in final report
|   ├── runs/                                                       # A sweep over model type, noise type, and noise strength
|   |   ├── P4CNN_aniso_std0.1_gamma2_lr0.002                       # a specific training run
|   |   |   ├── model.pt                                            # best model checkpoint
|   |   |   ├── run_data.json                                       # model training run_data
|   |   |   └── test_data.json                                      # test loss and accuracy
|   |   ├── ...
|   |   |
|   |   └── RelaxedP4CNN_none_std0_gamma0_lr0.002
|   └── plots/
|       ├── aniso_loss_plots.png
|       └── ...
|
├── MM_DD_YY/                                                       # any additional experiments, by date
|   └── ...                                                         # same strucutre as production
|
└── README.md                                                       # This documentation

