import pandas as pd
import config
from utils import visualize_data_distribution, load_metadata, generate_all_test_figures
from qa_utils import run_qa_process
from train import run_training, baseline_training 
from test import run_test_evaluation, run_test_visualization, run_single_inference, load_test_data, load_best_model
import os

def main():
    metadata_path = config.METADATA_PATH
    df = load_metadata(metadata_path)
    if df is None or df.empty:
        return

    while True:
        print("\n" + "="*50)
        print("RAT THERMAL SEGMENTATION - PROJECT MENU")
        print("="*50)
        print("1. Run Data QA and Cleaning")
        print("2. Visualize Image Distribution")
        print("3. Run BASELINE Training")
        print("4. Run K-FOLD Training")
        print("5. Run Test Evaluation (Metrics + Figures)")
        print("6. Run Test Visualization (Random Sample)")
        print("7. Run Single Inference & Visualize")
        print("8. RUN FULL PIPELINE (Train All -> Test All)")
        print("9. Print Model Architecture")
        print("10. Exit")
        print("="*50)

        choice = input(">>> Select option: ").strip()

        if choice == '1':
            run_qa_process(df, metadata_path)
        elif choice == '2':
            visualize_data_distribution(df)
        elif choice == '3':
            baseline_training(df)
        elif choice == '4':
            run_training(df)

        if choice in ['5','6']:
            model_choice = input("Select model type for test (baseline/kfolds): ").strip().lower()
            
            if model_choice == 'baseline':
                # Pass 'baseline' to load_test_data so it finds the correct JSON
                test_loader, df_test, test_dataset = load_test_data(df, model_type='baseline')
                if test_loader is None: continue

                if choice == '5':
                    run_test_evaluation(df, model_type='baseline')
                    model = load_best_model('baseline')
                    if model:
                        generate_all_test_figures(test_dataset, model, 'baseline')
                else:
                    model = load_best_model('baseline')
                    run_test_visualization(df, model_type='baseline')

            elif model_choice == 'kfolds':
                try:
                    fold_input = int(input(f"Enter fold number (1-{config.K_FOLDS}): ").strip())
                    # Pass 'kfolds' and the specific fold to load_test_data
                    test_loader, df_test, test_dataset = load_test_data(df, model_type='kfolds', fold=fold_input)
                    if test_loader is None: continue

                    if choice == '5':
                        run_test_evaluation(df, model_type='kfolds', fold=fold_input)
                        model = load_best_model('kfolds', fold=fold_input)
                        if model:
                            generate_all_test_figures(test_dataset, model, 'kfolds', fold_num=fold_input)
                    else:
                        run_test_visualization(df, model_type='kfolds', fold=fold_input)
                except ValueError:
                    print("[ERROR] Invalid fold input.")

        elif choice == '7':
            m_type = input("Model type (baseline/kfolds): ").strip().lower()
            fold = int(input("Fold (or 0 for baseline): ")) if m_type == 'kfolds' else 1
            run_single_inference(m_type, fold)

        elif choice == '8':
            print("\n>>> STARTING FULL PIPELINE...")
            # 1. Baseline
            baseline_training(df)
            run_test_evaluation(df, 'baseline')
            # 2. K-Folds
            run_training(df)
            for f in range(1, config.K_FOLDS + 1):
                run_test_evaluation(df, 'kfolds', fold=f)
            print("\n>>> PIPELINE FINISHED.")

        elif choice == '9':
            from model import build_model
            print(build_model())

        elif choice == '10':
            break

if __name__ == "__main__":
    main()