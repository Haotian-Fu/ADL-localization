import matplotlib.pyplot as plt

def plot_accuracy_per_session(sessions, accuracy_dict):
    """
    Plot a line chart: x-axis = iteration, y-axis = accuracy
    for each session.

    Args:
      sessions: list of session names (strings),
                e.g. ["SB-94975U", "0exwAT_ADL_1", ...]
      accuracy_dict: dict keyed by session name, 
                     each value is a list (or array) of accuracy values 
                     for iteration in 0..N-1.
                     e.g. accuracy_dict["SB-94975U"] = [0.84, 0.86, 0.88, ...]

    Example usage:
      sessions = ["S1", "S2"]
      accuracy_dict = {
          "S1": [0.8, 0.82, 0.85],
          "S2": [0.75, 0.78, 0.80]
      }
      plot_accuracy_per_session(sessions, accuracy_dict)
    """
    # 1) Prepare iteration array: 11 points
    iterations = list(range(0, 101, 10))  # => [0,10,20,30,40,50,60,70,80,90,100]
    
    # 2) Gather all accuracy data to compute min/max
    all_accuracies = []
    for session in sessions:
        # We expect each session's accuracy list to have length 11
        all_accuracies.extend(accuracy_dict[session])
    
    # 3) Compute y_min, y_max with the desired logic
    raw_min = min(all_accuracies)
    raw_max = max(all_accuracies)
    y_min = max(0, raw_min - 5)
    y_max = min(100, raw_max + 5)

    # 4) Plot
    plt.figure(figsize=(8, 5))
    
    for session in sessions:
        acc_list = accuracy_dict[session]
        if len(acc_list) != len(iterations):
            print(f"Warning: session {session} has {len(acc_list)} data points, "
                  f"but iteration array has {len(iterations)}. They must match.")
        # Plot each session's curve
        plt.plot(iterations, acc_list, marker='o', label=session)

    # 5) Configure axes, legend, etc.
    plt.xlabel("Iteration (0..100 step=10)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Iteration per Session")
    plt.ylim(y_min, y_max)  # clamp using the logic from your requirement
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
# Example usage:
if __name__ == "__main__":
    sessions = ["0exwAT_ADL_1"]
    accuracy_dict = {
        "0exwAT_ADL_1": [51.60, 51.12, 51.08, 51.12, 51.12, 51.08, 51.12, 51.12, 51.08, 51.12, 51.12]
    }
    plot_accuracy_per_session(sessions, accuracy_dict)