import argparse
import random
import sys
from timeit import default_timer as timer

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from utils import data_setup, model, engine, utils, loss_function, data_preprocess, augmentation_utils

if __name__ == "__main__":
    # setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="01_Cypriniformes.fasta", type=str,
                        help="choose a fasta file in data directory")
    parser.add_argument("--k", default=6, type=int,
                        help="k-mer size, an integer between 6-8")
    parser.add_argument("--weak_mutation_rate", default=1e-4, type=float,
                        help="Weak mutation rate for augmented data.")
    parser.add_argument("--strong_mutation_rate", default=1e-2, type=float,
                        help="Strong mutation rate for augmented data.")
    parser.add_argument("--weak_fragmentation_perc", default=None, type=float,
                        help="Weak fragmentation percentage for augmented data.")
    parser.add_argument("--strong_fragmentation_perc", default=None, type=float,
                        help="Strong fragmentation percentage for augmented data.")
    parser.add_argument("--number_of_pairs", default=1, type=int,
                        help="Number of augmented data pairs to generate.")
    parser.add_argument("--number_of_models", default=5, type=int,
                        help="number of models")
    parser.add_argument("--lr", default=7e-5, type=float,
                        help="learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="weight decay")
    parser.add_argument("--temp_ins", default=0.1, type=float,
                        help="instance temperature")
    parser.add_argument("--temp_clu", default=1.0, type=float,
                        help="cluster temperature")
    parser.add_argument("--num_epochs", default=100, type=int,
                        help="number of epochs")
    parser.add_argument("--batch_size", default=512, type=int,
                        help="batch size")
    parser.add_argument("--embedding_dim", default=512, type=int,
                        help="embedding dimension")
    parser.add_argument("--feature_dim", default=128, type=int,
                        help="feature dimension")
    parser.add_argument("--random_seed", default=0)
    parser.add_argument("--weight", default=0.7)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    if args.output_file is not None:
        sys.stdout = open("../results/" + args.output_file, "w")
    print(args)

    ####################################################################################################################
    print("Reading the data...")
    records_df = data_preprocess.read_fasta("data/" + args.dataset)
    class_names = sorted(records_df.label.unique())
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    ####################################################################################################################
    # Generate Augmented Data Pairs
    print("Generating Augmented Data Pairs...")
    random.seed(42)
    np.random.seed(42)
    if args.weak_mutation_rate is not None:
        X_train, X_test, y_test = augmentation_utils.generate_pairs(
            data=records_df,
            class_to_idx=class_to_idx,
            k=args.k,
            number_of_pairs=args.number_of_pairs,
            mutation_rate_weak=args.weak_mutation_rate,
            mutation_rate_strong=args.strong_mutation_rate
        )
    elif args.weak_fragmentation_perc is not None:
        X_train, X_test, y_test = augmentation_utils.generate_pairs(
            data=records_df,
            class_to_idx=class_to_idx,
            k=args.k,
            number_of_pairs=args.number_of_pairs,
            frag_perc_weak=args.weak_fragmentation_perc,
            frag_perc_strong=args.strong_fragmentation_perc
        )
    else:
        raise ValueError("Specify either mutation rates or fragmentation percentages for augmentation.")

    # data normalization
    X_train, X_test = utils.data_normalization(X_train, X_test)
    print(f"Class names: {class_names}")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape} | Number of labels in y_test: {len(y_test)}")

    ####################################################################################################################
    # Create Datasets and DataLoaders
    random.seed(args.random_seed)
    NUM_WORKERS = 1

    train_data = data_setup.PairSeqData(train_pairs=X_train,
                                        transform=None)
    test_data = data_setup.SeqData(sequences=X_test,
                                   labels=y_test,
                                   classes=class_names,
                                   class_to_idx=class_to_idx,
                                   transform=None)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  num_workers=NUM_WORKERS,
                                  drop_last=False,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=args.batch_size,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)
    ####################################################################################################################
    li = np.linspace(0.0, 1.0, 11)
    result_weight_hard = []
    result_weight_soft = []
    for weight in li:
        print(f"Weight: {weight}")
        y_preds = []
        y_probs = []
        for i in range(args.number_of_models):
            # Training
            print(f"Training model #{i + 1}")
            torch.manual_seed(args.random_seed + i)
            torch.cuda.manual_seed(args.random_seed + i)
            # initialize the model
            backbone_model = model.BackBoneModel(input_shape=1,
                                                 output_shape=args.embedding_dim)
            # backbone_model = model.get_resnet("ResNet18")
            projector_model = model.Network(backbone=backbone_model,
                                            rep_dim=args.embedding_dim,
                                            feature_dim=args.feature_dim,
                                            class_num=len(class_names)).to(device)
            # Setup loss function and optimizer
            optimizer = torch.optim.Adam(
                [
                    {"params": projector_model.backbone.parameters(), "lr": args.lr, },
                    {"params": projector_model.instance_projector.parameters(), "lr": args.lr},
                    {"params": projector_model.cluster_projector.parameters(), "lr": args.lr},
                ],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
            criterion_instance = loss_function.InstanceLoss(args.batch_size, args.temp_ins, device).to(device)
            criterion_cluster = loss_function.ClusterLoss(len(class_names), args.temp_clu, device).to(device)

            # start the timer
            start_time = timer()

            # train model
            model_results = engine.train(model=projector_model,
                                         train_dataloader=train_dataloader,
                                         test_dataloader=test_dataloader,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         weight=weight,
                                         criterion_instance=criterion_instance,
                                         criterion_cluster=criterion_cluster,
                                         epochs=args.num_epochs)

            # end the timer
            end_time = timer()
            total_time = (end_time - start_time)
            print(f"Total training time: {total_time:.3f} seconds")

            print(f"Evaluating model")
            y_prob, y_pred, ind, acc = engine.model_evaluation(model=projector_model, X_test=X_test, y_test=y_test)
            print(f"Accuracy of model: {acc * 100:.2f}%")
            # utils.plot_loss_curves(model_results, total_time=total_time)
            d = {}
            for j, k in ind:
                d[j] = k
            for j in range(len(y_pred)):  # we do this for each sample or sample batch
                y_pred[j] = d[y_pred[j]]
            y_preds.append(y_pred)
            y_prob_hungarian = np.zeros_like(y_prob)
            for j in range(len(d.keys())):  # we do this for each sample or sample batch
                y_prob_hungarian[:, d[j]] = y_prob[:, j]
            y_probs.append(y_prob_hungarian)
            print("#" * 100)

        ####################################################################################################################
        # Hard voting
        y_preds = np.array(y_preds)
        mode, counts = stats.mode(y_preds, axis=0)

        w = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for i in range(y_test.shape[0]):
            w[y_test[i], mode[i]] += 1
        print(
            f"Accuracy of hard voting of {args.number_of_models} models: {100 * np.sum(np.diag(w) / np.sum(w)):.2f}")
        print("Confusion matrix:")
        print(w)
        result_weight_hard.append(np.sum(np.diag(w) / np.sum(w)))

        ####################################################################################################################
        # Soft voting
        y_probs = np.array(y_probs)

        y_prob = []
        for i in range(y_probs.shape[1]):
            prob = np.zeros(y_probs.shape[2])
            for j in range(y_probs.shape[0]):
                prob += y_probs[j][i]
            prob /= (y_probs.shape[0])
            y_prob.append(prob)

        w = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for i in range(y_test.shape[0]):
            w[y_test[i], y_prob[i].argmax()] += 1
        print(
            f"Accuracy of soft voting of {args.number_of_models} models: {100 * np.sum(np.diag(w) / np.sum(w)):.2f}")
        print("Confusion matrix:")
        print(w)
        result_weight_soft.append(np.sum(np.diag(w) / np.sum(w)))

    print(result_weight_hard)
    print(result_weight_soft)
