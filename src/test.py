from evaluate import calculate_metrics, evaluate


def test(model, model_path, data_loader, device):
    accuracy, precision, recall, f1_score = calculate_metrics(model,
                                                              model_path,
                                                              data_loader,
                                                              device)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
