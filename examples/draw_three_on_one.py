import json  
import matplotlib.pyplot as plt  
  
def load_loss_data(file_path):  
    """Load loss data from a JSON file."""  
    with open(file_path, 'r') as f:  
        data = json.load(f)  
    return data 
  
def plot_loss_curves(loss_data_list, labels):  
    """Plot loss curves on the same plot."""  
    plt.figure(figsize=(10, 6))  
    for loss_data, label in zip(loss_data_list, labels):  
        plt.plot(loss_data, label=label)  
    plt.xlabel('Iteration')  
    plt.ylabel('Loss')  
    plt.title('Loss Curves Comparison')  
    plt.legend()  
    plt.grid(True)  
    plt.savefig('loss_curves_comparison.png')
  
def main():  
    # Load loss data from JSON files  
    loss_data1 = load_loss_data('FP16_loss_bs256.json')  
    loss_data2 = load_loss_data('O2_loss_bs256.json')  
    loss_data3 = load_loss_data('O2_FP8act_loss_bs256.json')  
  
    # Combine loss data into a list  
    loss_data_list = [loss_data1, loss_data2, loss_data3]  
    labels = ['FP16 MLP model', 'FP8 O2 MLP model', 'FP8 O2 + FP8 activation MLP model']  
  
    # Plot loss curves  
    plot_loss_curves(loss_data_list, labels)  
  
if __name__ == "__main__":  
    main()  
