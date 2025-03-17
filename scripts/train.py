# Function to create windowed data
# Example of trying different window sizes
# need to save the model at the end of training

window_sizes = [5, 10, 20]  # 100ms, 200ms, 400ms (assuming 20ms bins)

results = {}

for window_size in window_sizes:
    # Create windowed data
    X_windowed = create_windows(neural_data, window_size)
    y_windowed = create_windows(labels, window_size).mean(axis=2)

    # Train and evaluate model
    model = CNNSilenceDecoder(n_channels=256, window_size=window_size)


    results[window_size] = evaluate_model(model, X_test, y_test)