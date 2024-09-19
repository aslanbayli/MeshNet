import torch
import torch_geometric
import sys
sys.path.append('.')
from src.models.utils import f1_score, cbpe_confusion_matrix, create_mappings
from src.models.architectures.gnn import GraphNN
from src.data.make_dataset import load_graphs


def predict(model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    
    dataset = load_graphs([i for i in range(0, 4)], 100, True)

    data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, pair in enumerate(data_loader):
            batch = pair[0]
            indices = pair[1]

            batch = batch.to(device)

            output = model(batch).flatten()
            updated_output = output[:int(output.shape[0]/3*2)]

            # create_mappings(output, indices, idx, True)

            tp, fp, fn, tn = cbpe_confusion_matrix(updated_output)
            precision, recall, fscore = f1_score([tp, fp, fn, tn])

            print("\nTP: {:.4f}, FP: {:.4f}, FN: {:.4f}, TN: {:.4f}".format(tp, fp, fn, tn))
            print("Precision: {:.4f} % | Recall: {:.4f} %".format(precision, recall))
            print("Confidence estimate: {:.4f} %".format(fscore))


def main():
    epochs = 100
    lr = 1e-3
    bs = 5
    hs = 64

    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
        lr = float(sys.argv[2])
        bs = int(sys.argv[3])
        hs = int(sys.argv[4])

    model = GraphNN(100, 1, hs)
    model_path = f"./models/{epochs}_{lr}_{bs}_{hs}.pt"
    print('Model - Epochs: {}, Learning Rate: {}, Batch Size: {}, Hidden Size: {}\n'.format(epochs, lr, bs, hs))
    predict(model, model_path)


if __name__ == '__main__':
    main()
