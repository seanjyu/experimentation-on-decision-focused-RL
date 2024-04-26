import numpy as np
import matplotlib.pyplot as plt
seed_num = 11
file_starter = 'csv/DF_Q-learning_seed' + str(seed_num)

file_1 = np.loadtxt(file_starter + '_1.csv', delimiter=',', skiprows=1)  # Assuming CSV files have headers
file_2 = np.loadtxt(file_starter + '_2.csv', delimiter=',', skiprows=1)  # Assuming CSV files have headers
file_3 = np.loadtxt(file_starter + '_3.csv', delimiter=',', skiprows=1)  # Assuming CSV files have headers
file_4 = np.loadtxt(file_starter + '_4.csv', delimiter=',', skiprows=1)  # Assuming CSV files have headers
file_5 = np.loadtxt(file_starter + '_5.csv', delimiter=',', skiprows=1)  # Assuming CSV files have headers

test_soft_eval_1 = file_1[1:, 10]
test_soft_eval_2 = file_2[1:, 10]
test_soft_eval_3 = file_3[1:, 10]
test_soft_eval_4 = file_4[1:, 10]
test_soft_eval_5 = file_5[1:, 10]

test_strict_eval_1 = file_1[1:, 9]
test_strict_eval_2 = file_2[1:, 9]
test_strict_eval_3 = file_3[1:, 9]
test_strict_eval_4 = file_4[1:, 9]
test_strict_eval_5 = file_5[1:, 9]

test_loss_1 = file_1[1:, 8]
test_loss_2 = file_2[1:, 8]
test_loss_3 = file_3[1:, 8]
test_loss_4 = file_4[1:, 8]
test_loss_5 = file_5[1:, 8]

# plot test_soft_eval
test_soft_eval_combined = np.array([test_soft_eval_1, test_soft_eval_2, test_soft_eval_3, test_soft_eval_4, test_soft_eval_5])

# Calculate the mean and standard deviation across rows
test_soft_eval_mean = np.mean(test_soft_eval_combined, axis=0)
test_soft_eval_std_dev = np.std(test_soft_eval_combined, axis=0)

# Generate x values (assuming the length of the arrays is the same)
x_values = np.arange(1, len(test_soft_eval_mean) + 1)

# Plot the average line
plt.plot(x_values, test_soft_eval_mean)

# Plot the region between mean plus std deviation and mean minus std deviation
plt.fill_between(x_values, test_soft_eval_mean + 1.96 * test_soft_eval_std_dev, test_soft_eval_mean - 1.96 * test_soft_eval_std_dev, alpha=0.3)

# Add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Test Error')
plt.title('Test Soft Eval')
# plt.legend()

# Show the plot
plt.show()

# plot test_strict_eval
test_strict_eval_combined = np.array([test_strict_eval_1, test_strict_eval_2, test_strict_eval_3, test_strict_eval_4, test_strict_eval_5])

# Calculate the mean and standard deviation across rows
test_strict_eval_mean = np.mean(test_strict_eval_combined, axis=0)
test_strict_eval_std_dev = np.std(test_strict_eval_combined, axis=0)

# Generate x values (assuming the length of the arrays is the same)
x_values = np.arange(1, len(test_strict_eval_mean) + 1)

# Plot the average line
plt.plot(x_values, test_strict_eval_mean)

# Plot the region between mean plus std deviation and mean minus std deviation
plt.fill_between(x_values, test_strict_eval_mean + 1.96 * test_strict_eval_std_dev, test_strict_eval_mean - 1.96 * test_strict_eval_std_dev, alpha=0.3)

# Add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Eval')
plt.title('Test Strict Eval')
# plt.legend()

# Show the plot
plt.show()

# plot test_loss
test_loss_combined = np.array([test_loss_1, test_loss_2, test_loss_3, test_loss_4, test_loss_5])

# Calculate the mean and standard deviation across rows
test_loss_mean = np.mean(test_loss_combined, axis=0)
test_loss_std_dev = np.std(test_loss_combined, axis=0)

# Generate x values (assuming the length of the arrays is the same)
x_values = np.arange(1, len(test_loss_mean) + 1)

lower_bound_limit = np.maximum(0, test_loss_mean - test_loss_std_dev * 1.96)

# Plot the average line
plt.plot(x_values, test_loss_mean)

# Plot the region between mean plus std deviation and mean minus std deviation
# plt.fill_between(x_values, test_loss_mean + 1.96 * test_loss_std_dev, test_loss_mean - 1.96 * test_loss_std_dev, alpha=0.3)
plt.fill_between(x_values, test_loss_mean + 1.96 * test_loss_std_dev, lower_bound_limit, alpha=0.3, where=test_loss_mean + 1.96 * test_loss_std_dev >= lower_bound_limit)

# Add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss')
# plt.legend()

# Show the plot
plt.show()