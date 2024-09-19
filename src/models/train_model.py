import torch_geometric
import torch
import sys
sys.path.append('.')
from src.data.make_dataset import load_graphs
from src.models.utils import accuracy, confusion_matrix, f1_score, create_mappings
from src.models.architectures.gnn import GraphNN
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import copy


def validate(model, validation, flag, hh_num, epoch, detailed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # switch to evaluation mode
    val_fscore = 0
    val_loss = 0

    with torch.no_grad():
        if not flag:
            if (epoch+1) % 10 == 0 or epoch == 0 or detailed:
                print("\n### Validation ###")
                
        for pair in validation:
            batch = pair[0]
            indices = pair[1]
            batch = batch.to(device)

            predictions = model(batch).flatten()
            
            # balance the dataset with weights
            w = compute_class_weight(class_weight='balanced', classes=[0,1], y=batch.y.cpu().numpy())
            w = torch.where(batch.y == 1, w[1], w[0])
            # calculate loss
            criterion = torch.nn.BCELoss(reduction='mean', weight=w).to(device)
            # criterion = torch.nn.BCELoss(reduction='mean').to(device)
            loss = criterion(predictions, batch.y)
            val_loss += loss.item()

            pos_acc, neg_acc = accuracy(model, predictions, batch.y)

            tp, fp, fn, tn = confusion_matrix(predictions, batch.y)
            precision, recall, fscore = f1_score([tp, fp, fn, tn])
            val_fscore += fscore

            # print loss and accuracy
            if not flag:
                if (epoch+1) % 10 == 0 or epoch == 0 or detailed:
                    print("Loss: {:.4f} | Positive Edge Accuracy: {:.4f} % | Negative Edge Accuracy: {:.4f} % | TP: {:.4f}, FP: {:.4f}, FN: {:.4f}, TN: {:.4f}".format(loss.item(), pos_acc, neg_acc, tp, fp, fn, tn))
                    print("Precision: {:.4f} % | Recall: {:.4f} %".format(precision, recall))

            # only do this when the training stops
            if flag:
                create_mappings(predictions, indices, hh_num, False)

        val_fscore /= len(validation)
        val_loss /= len(validation)
    

    model.train() # switch back to training mode

    return  val_fscore, val_loss


def train(model, epochs, learning_rate, batch_size, hidden_size, detailed=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = load_graphs([1,3,6,7,8,9,10], 100)
    v_hh = 2
    validation = load_graphs([v_hh], 100)

    # create data loader for batch training
    train_set = torch_geometric.loader.DataLoader(train, batch_size=batch_size, shuffle=False)
    # create validation data loader
    val_set = torch_geometric.loader.DataLoader(validation, batch_size=1)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # mode the models to the GPU if available
    model = model.to(device)

    # switch to training mode
    model.train() 

    count_fscore_decline = 0
    max_validation_fscore = 0
    max_fscore = 0
    flag = False
    train_loss = []
    val_loss = []
    train_fscore = []
    val_fscore = []

    print("Epochs: {}, Learning Rate: {}, Batch Size: {}, Hidden Size: {}\n".format(epochs, learning_rate, batch_size, hidden_size))
    
    for epoch in range(epochs):
        if flag:
            validate(model, val_set, flag, v_hh, epoch, detailed)
            break

        if epoch == epochs-1:
            flag = True

        matrix = [0,0,0,0]

        if epoch == 0 and not detailed:
            print("Epoch {}/{} ".format(epoch+1, epochs))
            print("### Training ###")

        if (epoch+1) % 10 == 0 or detailed:
            print("\nEpoch {}/{} ".format(epoch+1, epochs)) 
            print("### Training ###")

        epoch_train_loss = 0
        for pair in train_set:
            batch = pair[0]

            optimizer.zero_grad() # clear gradients 

            # move data to GPU if available
            batch = batch.to(device)

            # forward pass
            predictions = model(batch).flatten()
            # balance the dataset with weights
            w = compute_class_weight(class_weight='balanced', classes=[0,1], y=batch.y.cpu().numpy())
            w = torch.where(batch.y == 1, w[1], w[0])

            # calculate loss
            criterion = torch.nn.BCELoss(reduction='mean', weight=w).to(device)
            # criterion = torch.nn.BCELoss(reduction='mean').to(device)
            loss = criterion(predictions, batch.y)

            epoch_train_loss += loss.item()
            
            # calculate accuracy
            pos_acc, neg_acc = accuracy(model, predictions, batch.y)

            tp, fp, fn, tn = confusion_matrix(predictions, batch.y)
            matrix[0] += tp
            matrix[1] += fp
            matrix[2] += fn
            matrix[3] += tn
            
            # backward pass
            loss.backward()
            optimizer.step()

            # print loss and accuracy
            if (epoch+1) % 10 == 0 or epoch == 0 or detailed:
                print("Loss: {:.4f} | Positive Edge Accuracy: {:.4f} % | Negative Edge Accuracy: {:.4f} % | TP: {:.4f}, FP: {:.4f}, FN: {:.4f}, TN: {:.4f}".format(loss.item(), pos_acc, neg_acc, tp, fp, fn, tn))
        
        epoch_train_loss /= len(train_set)

        # calculate f-score for training
        precision, recall, fscore = f1_score(matrix)
        train_fscore.append(fscore)

        if (epoch+1) % 10 == 0 or epoch == 0 or detailed:
            print("Precision: {:.4f} % | Recall: {:.4f} %".format(precision, recall))
            print("F1-score: {:.4f} %".format(fscore))
            
        # calculate f-score for validation
        v_fscore, epoch_val_loss = validate(model, val_set, flag, v_hh, epoch, detailed) 
        val_fscore.append(v_fscore)

        scheduler.step(epoch_val_loss)
        
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        
        if (epoch+1) % 10 == 0 or epoch == 0 or detailed:
            print("Validation F1-score: {:.4f} %".format(v_fscore))
            print("-"*50)
        
        decline_threshold = 50
        if count_fscore_decline == decline_threshold:
            print(f"\nF1-score has declined for {decline_threshold} epochs, stopping training...")
            flag = True

        if int(max_validation_fscore) - int(v_fscore) >= 3:
            count_fscore_decline += 1  

        if (v_fscore > max_validation_fscore):
            max_validation_fscore = v_fscore
            max_fscore = fscore
            # save model
            torch.save(model.state_dict(), "./models/{}_{}_{}_{}.pt".format(epochs, learning_rate, batch_size, hidden_size))


    print("\nFinal F1-score: {:.4f} %".format(max_fscore))
    print("\nFinal Validation F1-score: {:.4f} %".format(max_validation_fscore))
    print("-"*100)
    print()

    torch.cuda.empty_cache()

    # plot epoch against loss
    epoch_list = [i for i in range(1, len(train_loss)+1)]
    plt.plot(epoch_list, train_loss, label="Training Loss")
    plt.plot(epoch_list, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./reports/loss.png")
    plt.clf()

    # plot epoch against f-score
    plt.plot(epoch_list, train_fscore, label="Training F1-score")
    plt.plot(epoch_list, val_fscore, label="Validation F1-score")
    plt.xlabel("Epochs")
    plt.ylabel("F1-score")
    plt.legend()
    plt.savefig(f"./reports/f1_score.png")
    plt.clf()

    # save maximum accuracy to a file for plotting
    with open(f"./reports/accuracy.csv", "a") as f:
        f.write("{}, {}, {}, {}, {:.4f} %, {:.4f} %\n".format(epochs, learning_rate, batch_size, hidden_size, max_fscore, max_validation_fscore))


def main():
    # with open("./reports/accuracy.csv", "w") as f:
    #     f.write("Epochs, Learning Rate, Batch Size, Hidden Size, F1-score, Validation F1-score\n")

    # for e in [300]:
    #     for lr in [1e-3, 1e-4, 1e-5]:
    #         for bs in [1, 3, 7]:
    #             for hs in [32, 64, 128]:
    #                 for dp in [0.05]:
    #                     model = GraphNN(100, 1, hidden_size=hs, dropout_prob=dp)
    #                     train(model, epochs=e, learning_rate=lr, batch_size=bs, hidden_size=hs, detailed=False)

    hs = 64
    dp = 0.05
    model = GraphNN(node_attr_size=100, edge_attr_size=1, hidden_size=hs, dropout_prob=dp)
    train(model, epochs=300, learning_rate=1e-3, batch_size=1, hidden_size=hs, detailed=False)



if __name__ == "__main__":
    main()