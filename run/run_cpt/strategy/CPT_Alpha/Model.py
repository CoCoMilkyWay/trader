from misc.pytorch_model import \
    ScalingMethod, SplitMethod, \
    DataCheckResult, ModelType, GeneralizedModel, \
    CNN, Recurrent, Transformer, Ensemble
    
def train(
    df,
    scaling_methods,
):
    feature_names = []
    label_names = []
    for col in df.columns:
        if 'label' in col:
            label_names.append(col)
        else:
            feature_names.append(col)
            
    TRAIN = "Ensemble"

    if TRAIN == "BiLSTM":
        model = GeneralizedModel(
            # Core Architecture Parameters
            model_type='bilstm',                # BiLSTM chosen for temporal pattern recognition
            input_dims=[len(feature_names)],       # Must match your feature count exactly
            output_dims=[len(label_names)],        # Must be 5 for your 5 future returns
            scaling_methods=scaling_methods,    # Defined in feature_specs for each feature
            # Cross-validation Parameters
            split_method=SplitMethod.KFOLD_CV,  # Regular K-fold since:
                                                # - This is regression, not classification (so no StratifiedKFold)
                                                # - Technical indicators already encode temporal info (so no TimeSeriesSplit)
            n_splits=2,                         # Standard value for k-fold CV:
                                                # - 5-10 splits is common practice
                                                # - 5 gives good balance of bias/variance
                                                # - Each fold has 20% validation data
            # BiLSTM Architecture Parameters
            hidden_size=64,                     # Determined by:
                                                # 1. Rule: 1.5-2x number of features
                                                # 2. Round to nearest power of 2 (32,64,128)
                                                # 3. If bidirectional=True, can use smaller size
            num_layers=2,                       # Typical value for most applications:
                                                # - First layer: Learn basic patterns
                                                # - Second layer: Learn feature interactions
                                                # - More layers rarely improve performance
            dropout=0.3,                        # Common ranges: 0.2-0.5
                                                # - Higher (0.3) because many technical features
                                                # - Helps prevent overfitting from feature correlation
            bidirectional=True,                 # True because:
                                                # - Technical patterns might be relevant in both directions
                                                # - Doubles the effective hidden size
        )
        # Training Parameters
        training_history = model.fit(
            X=df[feature_names].copy(),
            y=df[label_names].copy(),
            batch_size=32,                  # Small-medium batch size because:
                                            # - Too small (<8): unstable gradients
                                            # - Too large (>32): might miss patterns
                                            # - 16-32 is good range for most cases
            epochs=100,                     # Upper limit for training:
                                            # - Early stopping will prevent overfitting
                                            # - Complex relationships need time to learn
                                            # - Can be higher with early stopping
            early_stopping_patience=10,     # Monitor 10 epochs of no improvement:
                                            # - Too low (<5): might stop too early
                                            # - Too high (>20): wastes training time
                                            # - 10 is good balance for this complexity
            learning_rate=0.001             # Standard Adam learning rate:
                                            # - 0.001 is default for Adam optimizer
                                            # - Well-tested for most deep learning tasks
                                            # - Can try 0.0001-0.01 range if needed
        )
        model.save('models/bilstm')
        # # Make predictions
        # new_data = pd.DataFrame(
        #     np.random.randn(5, self.n_features),
        #     columns=[f'feature_{i}' for i in range(self.n_features)]
        # )
        # predictions = model.predict_single(new_data) # type: ignore
        # Save the model
        # model.save('bilstm_model')
        # Load the model later
        # loaded_model = GeneralizedModel.load('bilstm_model')
    elif TRAIN == "Ensemble":
        # Initialize model with XGBoost
        model = GeneralizedModel(
            model_type='xgboost',
            input_dims=[len(feature_names)],
            output_dims=[1],
            scaling_methods=scaling_methods,
            split_method=SplitMethod.RANDOM,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )
        # Train model
        history = model.fit(
            X=df[feature_names].copy(),
            y=df[['label_1']].copy(),
                           validation_split=0.2,
                           early_stopping_patience=10)
        model.save('models/xgboost')

        # # Make predictions
        # predictions = model.predict_single(X_test)