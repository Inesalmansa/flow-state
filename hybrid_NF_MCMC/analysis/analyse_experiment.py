import argparse
from my_workspace.HMC_NF.analysis.utils_analysis import load_config, list_model_files
import plot_rdfs
import plot_mean_x
import plot_p_acc_hr

def parse_args():
    parser = argparse.ArgumentParser(description="Run analyses on an experiment.")
    parser.add_argument('experiment_folder', type=str, help="Path to the experiment folder.")
    parser.add_argument('--analysis', type=str, choices=['rdfs', 'mean_x', 'p_acc_hr', 'all'],
                        default='all', help="Type of analysis to run (default: all).")
    parser.add_argument('--model_interval', type=int, default=1,
                        help="Interval for selecting models (e.g. every nth model to analyze)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.experiment_folder)
    model_files = list_model_files(args.experiment_folder)

    # Create a folder named 'analysis' within the experiment folder to store all plots
    import os
    analysis_folder = os.path.join(args.experiment_folder, "analysis")
    os.makedirs(analysis_folder, exist_ok=True)

    if args.analysis in ['rdfs', 'all']:
        print("Running RDF analysis...")
        plot_rdfs.run_rdf_analysis(analysis_folder, model_files, model_interval=args.model_interval)

    if args.analysis in ['mean_x', 'all']:
        print("Running mean x position analysis...")
        plot_mean_x.run_mean_x_analysis(analysis_folder)

    if args.analysis in ['p_acc_hr', 'all']:
        print("Running acceptance probability analysis...")
        plot_p_acc_hr.run_p_acc_hr_analysis(analysis_folder, model_files, model_interval=args.model_interval)

if __name__ == "__main__":
    main()