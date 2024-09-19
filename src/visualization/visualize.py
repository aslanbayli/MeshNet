import matplotlib.pyplot as plt
import sys

def plot_config(path):
    config = [] # corresponds to the line number -1
    acc = [] # corresponds to the last column value (F1-Score)

    # open the text file for reading
    with open(path, 'r') as file:
        # read the lines, skipping the first line
        lines = file.readlines()[1:]

        for line_number, line in enumerate(lines, 1):
            # split the line by comma
            parts = line.strip().split(',')

            # extract the last column value and convert it to a float
            last_column_value = float(parts[-1].strip('%'))  # remove '%' and convert to float

            # append line number and last column value to the respective lists
            config.append(line_number)
            acc.append(last_column_value)

    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(config, acc, marker='o')
    plt.xlabel('Configuration')
    plt.ylabel('F1-Score')
    plt.title('Configuration vs F1-Score')
    plt.grid(True)

    max_x = config[acc.index(max(acc))]
    max_y = max(acc)
    plt.annotate(
                    text=f'Best Configuration (x: {max_x}, y: {max_y})', 
                    xy=(max_x, max_y), 
                    xytext=(max_x, max_y+1),
                    arrowprops=dict(facecolor='red', headlength=5, headwidth=8),
                )
    
    plt.vlines([max_x], 0, [max_y], linestyle='dashed', colors='red')
    plt.hlines([max_y], 0, [max_x], linestyle='dashed', colors='red')

    plt.xlim(0,None)
    plt.ylim(0,None)
    # save the plot
    plt.savefig('./reports/best_config_5.png')


def pie_chart(path, line):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Predicted accurately', 'Predicted inaccurately'
    results = []
    with open(path, 'r') as file:
        # read the lines, skipping the first line
        lines = file.readlines()
        parts = lines[int(line)-1].strip().split(',')
        results.append(float(parts[-1].strip('%')))
        results.append(100 - float(parts[-1].strip('%')))

    _, ax1 = plt.subplots()
    plt.rcParams['figure.figsize'] = [40, 20]
    plt.rcParams["figure.autolayout"] = True
    ax1.pie(results, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, textprops={'fontsize': 12})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Pie chart of prediction accuracy', fontsize=14)
    plt.savefig('./reports/pie_chart_5.png')


if __name__ == '__main__':
    op = sys.argv[1]
    if op == 'cf':
        plot_config('./reports/accuracy4.csv')
    elif op == 'pie':
        line = sys.argv[2]
        pie_chart('./reports/accuracy4.csv', line)
    else:
        print('Invalid option')