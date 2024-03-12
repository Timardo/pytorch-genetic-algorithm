library(ggplot2)

gpu_only = read.csv("output/tensor_output_gpu_only.csv")

gpu_anomaly = ggplot(gpu_only, aes(x=population, y=time_seconds)) +
  geom_line(aes(y = time_seconds, colour = "GPU"), linewidth = 1.25) +
  labs(title = "\nGPU execution times for different populations showing an anomaly between ~20-30k population\n", x = "Population", y = "Time in seconds", color = "Device") +
  scale_color_manual(values=c("#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

gpu_anomaly
ggsave(file="output/img/gpu_anomaly.svg", plot=gpu_anomaly, width=10, height=8)

##############################################################################################################################################################################
##############################################################################################################################################################################

gpu_cpu = read.csv("output/tensor_output.csv")
cpu_subset = subset(gpu_cpu, device == "cpu")
gpu_subset = subset(gpu_cpu, device == "cuda" & population <= max(cpu_subset$population))
cpu_subset = aggregate(time_seconds ~ population, cpu_subset, mean)
gpu_subset = aggregate(time_seconds ~ population, gpu_subset, mean)

data_frame = data.frame(
  population = cpu_subset$population,
  time_cuda = gpu_subset$time_seconds,
  time_cpu = cpu_subset$time_seconds
)

breaks = c(25, 50, 75, 100, 200, 300, 400, 500, 600, 700)

gpu_cpu_comparison = ggplot(data_frame, aes(x=population)) + 
  geom_line(aes(y = time_cpu, colour = "CPU"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu, colour = "CPU"), size = 3) +
  geom_line(aes(y = time_cuda, colour = "GPU"), linewidth = 1.25) +
  geom_point(aes(y = time_cuda, colour = "GPU"), size = 3) +
  scale_y_log10(breaks = breaks) +
  scale_x_log10(breaks = c(500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000)) +
  labs(title = "\nGPU vs CPU execution times for different populations\n", x = "Population", y = "Time in seconds", color = "Device") +
  scale_color_manual(values=c("#CC6666", "#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

gpu_cpu_comparison
ggsave(file="output/img/gpu_cpu_comparison.svg", plot=gpu_cpu_comparison, width=10, height=8)

##############################################################################################################################################################################

gpu_cpu_comparison_nolog = ggplot(data_frame, aes(x=population)) + 
  geom_line(aes(y = time_cpu, colour = "CPU"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu, colour = "CPU"), size = 3) +
  geom_line(aes(y = time_cuda, colour = "GPU"), linewidth = 1.25) +
  geom_point(aes(y = time_cuda, colour = "GPU"), size = 3) +
  #scale_y_log10(breaks = breaks) +
  #scale_x_log10(breaks = c(500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000)) +
  labs(title = "\nGPU vs CPU execution times for different populations\n", x = "Population", y = "Time in seconds", color = "Device") +
  scale_color_manual(values=c("#CC6666", "#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

gpu_cpu_comparison_nolog
ggsave(file="output/img/gpu_cpu_comparison_nolog.svg", plot=gpu_cpu_comparison_nolog, width=10, height=8)

##############################################################################################################################################################################

gpu_cpu_comparison_logy = ggplot(data_frame, aes(x=population)) + 
  geom_line(aes(y = time_cpu, colour = "CPU"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu, colour = "CPU"), size = 3) +
  geom_line(aes(y = time_cuda, colour = "GPU"), linewidth = 1.25) +
  geom_point(aes(y = time_cuda, colour = "GPU"), size = 3) +
  scale_y_log10(breaks = breaks) +
  #scale_x_log10(breaks = c(500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000)) +
  labs(title = "\nGPU vs CPU execution times for different populations\n", x = "Population", y = "Time in seconds", color = "Device") +
  scale_color_manual(values=c("#CC6666", "#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

gpu_cpu_comparison_logy
ggsave(file="output/img/gpu_cpu_comparison_logy.svg", plot=gpu_cpu_comparison_logy, width=10, height=8)

##############################################################################################################################################################################
##############################################################################################################################################################################

generations_output = read.csv("output/tensor_output_generations.csv")
generations_cpu = subset(generations_output, device == "cpu")
generations_gpu = subset(generations_output, device == "cuda" & generations <= max(generations_cpu$generations))

generations_gpu_plot = ggplot(generations_gpu, aes(x=generations, y=time_seconds)) +
  geom_line(aes(y = time_seconds, colour = "GPU"), linewidth = 1.25) +
  geom_point(aes(y = time_seconds, colour = "GPU"), size = 3) +
  labs(title = "\nGPU execution times for different generations showing near perfect linear growth\n", x = "Generations", y = "Time in seconds", color = "Device") +
  scale_color_manual(values=c("#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

generations_gpu_plot
ggsave(file="output/img/generations_gpu.svg", plot=generations_gpu_plot, width=10, height=8)
##############################################################################################################################################################################

generations_cpu_plot = ggplot(generations_cpu, aes(x=generations, y=time_seconds)) +
  geom_line(aes(y = time_seconds, colour = "CPU"), linewidth = 1.25) +
  geom_point(aes(y = time_seconds, colour = "CPU"), size = 3) +
  labs(title = "\nCPU execution times for different generations showing near perfect linear growth\n", x = "Generations", y = "Time in seconds", color = "Device") +
  scale_color_manual(values=c("#CC6666")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

generations_cpu_plot
ggsave(file="output/img/generations_cpu.svg", plot=generations_cpu_plot, width=10, height=8)


##############################################################################################################################################################################
##############################################################################################################################################################################

knapsack_size_output = read.csv("output/tensor_output_knapsack_size.csv")
knapsack_size_cpu = subset(knapsack_size_output, device == "cpu")
knapsack_size_gpu = subset(knapsack_size_output, device == "cuda" & knapsack_size <= max(knapsack_size_cpu$knapsack_size))

data_frame_knapsack = data.frame(
  knapsack_size = knapsack_size_cpu$knapsack_size,
  time_cuda = knapsack_size_gpu$time_seconds,
  time_cpu = knapsack_size_cpu$time_seconds
)

knapsack_sizes_plot = ggplot(data_frame_knapsack, aes(x=knapsack_size)) + 
  geom_line(aes(y = time_cpu, colour = "CPU"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu, colour = "CPU"), size = 3) +
  geom_line(aes(y = time_cuda, colour = "GPU"), linewidth = 1.25) +
  geom_point(aes(y = time_cuda, colour = "GPU"), size = 3) +
  scale_y_log10() +
  scale_x_log10() +
  labs(title = "\nGPU vs CPU execution times for different knapsack sizes showin near identical results as with population\n", x = "Knapsack size in items", y = "Time in seconds", color = "Device") +
  scale_color_manual(values=c("#CC6666", "#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

knapsack_sizes_plot
ggsave(file="output/img/knapsack_sizes.svg", plot=knapsack_sizes_plot, width=10, height=8)
