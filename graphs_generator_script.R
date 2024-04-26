library(ggplot2)

##############################################################################################################################################################################
##############################################################################################################################################################################

naive_knapsack = read.csv("output/naive_output_knapsack.csv")
naive_knapsack = aggregate(time_seconds ~ population, naive_knapsack, mean)

naive_knapsack_data_frame = data.frame(
  population = naive_knapsack$population,
  time = naive_knapsack$time_seconds
)

naive_knapsack_population = ggplot(naive_knapsack_data_frame, aes(x=population)) + 
  geom_line(aes(y = time), linewidth = 1.25) +
  geom_point(aes(y = time), size = 3) +
  labs(title = "\nZávislosť času behu od veľkosti populácie pri naivnej implementácii\n", x = "Populácia", y = "Čas (s)") +
  theme_minimal()

naive_knapsack_population
ggsave(file="output/img/naive_knapsack_population.svg", plot=naive_knapsack_population, width=10, height=8)

##############################################################################################################################################################################
##############################################################################################################################################################################

gpu_only = read.csv("output/tensor_output_gpu_only.csv")

gpu_anomaly = ggplot(gpu_only, aes(x=population, y=time_seconds)) +
  geom_line(aes(y = time_seconds, colour = "GPU"), linewidth = 1.25) +
  labs(title = "\nGraf anomálie pri spustení na GPU pri populácii okolo 20-30k\n", x = "Populácia", y = "Čas (s)", color = "Zariadenie") +
  scale_color_manual(values=c("#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

gpu_anomaly
ggsave(file="output/img/gpu_anomaly.svg", plot=gpu_anomaly, width=10, height=8)

##############################################################################################################################################################################
##############################################################################################################################################################################

gpu_cpu = read.csv("output/tensor_output.csv")
naive_bus = read.csv("output/naive_output_knapsack.csv")
naive_bus = aggregate(time_seconds ~ population, naive_bus, mean)
cpu_subset = subset(gpu_cpu, device == "cpu")
gpu_subset = subset(gpu_cpu, device == "cuda" & population <= max(cpu_subset$population))
cpu_subset = aggregate(time_seconds ~ population, cpu_subset, mean)
gpu_subset = aggregate(time_seconds ~ population, gpu_subset, mean)
cpu_naive_seconds = naive_bus$time_seconds
cpu_naive_seconds = c(cpu_naive_seconds, rep(NA, length(cpu_subset$population) - length(cpu_naive_seconds)))

data_frame = data.frame(
  population = cpu_subset$population,
  time_cuda = gpu_subset$time_seconds,
  time_cpu = cpu_subset$time_seconds,
  time_cpu_naive = cpu_naive_seconds
)

breaks = c(25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200)

gpu_cpu_comparison = ggplot(data_frame, aes(x=population)) + 
  geom_line(aes(y = time_cpu_naive, colour = "CPU Naive"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu_naive, colour = "CPU Naive"), size = 3) +
  geom_line(aes(y = time_cpu, colour = "CPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu, colour = "CPU Tensor"), size = 3) +
  geom_line(aes(y = time_cuda, colour = "GPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_cuda, colour = "GPU Tensor"), size = 3) +
  scale_y_log10(breaks = breaks) +
  scale_x_log10(breaks = c(500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000)) +
  labs(title = "\nZávislosť času behu od populácie pre rôzne implementácie úlohy o batohu\n", x = "Populácia", y = "Čas (s)", color = "Implementácia") +
  scale_color_manual(values=c("#CC66FF", "#CC6666", "#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

gpu_cpu_comparison
ggsave(file="output/img/gpu_cpu_comparison.svg", plot=gpu_cpu_comparison, width=10, height=8)

##############################################################################################################################################################################

gpu_cpu_comparison_nolog = ggplot(data_frame, aes(x=population)) + 
  geom_line(aes(y = time_cpu_naive, colour = "CPU Naive"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu_naive, colour = "CPU Naive"), size = 3) +
  geom_line(aes(y = time_cpu, colour = "CPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu, colour = "CPU Tensor"), size = 3) +
  geom_line(aes(y = time_cuda, colour = "GPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_cuda, colour = "GPU Tensor"), size = 3) +
  #scale_y_log10(breaks = breaks) +
  #scale_x_log10(breaks = c(500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000)) +
  labs(title = "\nZávislosť času behu od populácie pre rôzne implementácie úlohy o batohu\n", x = "Populácia", y = "Čas (s)", color = "Implementácia") +
  scale_color_manual(values=c("#CC66FF", "#CC6666", "#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

gpu_cpu_comparison_nolog
ggsave(file="output/img/gpu_cpu_comparison_nolog.svg", plot=gpu_cpu_comparison_nolog, width=10, height=8)

##############################################################################################################################################################################

gpu_cpu_comparison_logy = ggplot(data_frame, aes(x=population)) + 
  geom_line(aes(y = time_cpu_naive, colour = "CPU Naive"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu_naive, colour = "CPU Naive"), size = 3) +
  geom_line(aes(y = time_cpu, colour = "CPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu, colour = "CPU Tensor"), size = 3) +
  geom_line(aes(y = time_cuda, colour = "GPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_cuda, colour = "GPU Tensor"), size = 3) +
  scale_y_log10(breaks = breaks) +
  #scale_x_log10(breaks = c(500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000)) +
  labs(title = "\nZávislosť času behu od populácie pre rôzne implementácie úlohy o batohu\n", x = "Populácia", y = "Čas (s)", color = "Implementácia") +
  scale_color_manual(values=c("#CC66FF", "#CC6666", "#9999CC")) +
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
  geom_line(aes(y = time_seconds, colour = "GPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_seconds, colour = "GPU Tensor"), size = 3) +
  labs(title = "\nZávislosť času paralelnej implementácie na GPU od počtu generácií zobrazujúca takmer perfektnú lienárnu závislosť\n", x = "Generácie", y = "Čas (s)", color = "Implementácia") +
  scale_color_manual(values=c("#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

generations_gpu_plot
ggsave(file="output/img/generations_gpu.svg", plot=generations_gpu_plot, width=10, height=8)
##############################################################################################################################################################################

generations_cpu_plot = ggplot(generations_cpu, aes(x=generations, y=time_seconds)) +
  geom_line(aes(y = time_seconds, colour = "CPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_seconds, colour = "CPU Tensor"), size = 3) +
  labs(title = "\nZávislosť času paralelnej implementácie na CPU od počtu generácií zobrazujúca takmer perfektnú lienárnu závislosť\n", x = "Generácie", y = "Čas (s)", color = "Implementácia") +
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

##############################################################################################################################################################################
##############################################################################################################################################################################

tensor_bus = read.csv("output/tensor_output_bus.csv")
tensor_bus$time_seconds = tensor_bus$time_seconds / (tensor_bus$generations / 1000)
tensor_bus = aggregate(time_seconds ~ problem + device, tensor_bus, mean)
cpu_subset = subset(tensor_bus, device == "cpu")
cpu_other_subset = subset(tensor_bus, device == "cpu_other")
gpu_subset = subset(tensor_bus, device == "cuda")

tensor_bus_frame = data.frame(
  problem = tensor_bus$problem,
  time_cpu = cpu_subset$time_seconds,
  time_cpu_other = cpu_other_subset$time_seconds,
  time_gpu = gpu_subset$time_seconds
)

breaks = c(25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200)

tensor_bus_plot = ggplot(tensor_bus_frame, aes(x=problem)) + 
  geom_line(aes(y = time_cpu_other, colour = "CPU Java"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu_other, colour = "CPU Java"), size = 3) +
  geom_line(aes(y = time_cpu, colour = "CPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_cpu, colour = "CPU Tensor"), size = 3) +
  geom_line(aes(y = time_gpu, colour = "GPU Tensor"), linewidth = 1.25) +
  geom_point(aes(y = time_gpu, colour = "GPU Tensor"), size = 3) +
  scale_x_continuous(breaks = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) +
  labs(title = "\nČasová závislosť od riešenej úlohy na 1000 generácií\n", x = "Úloha", y = "Čas (s)", color = "Implementácia") +
  scale_color_manual(values=c("#CC66FF", "#CC6666", "#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

tensor_bus_plot
ggsave(file="output/img/tensor_bus_1.svg", plot=tensor_bus_plot, width=10, height=8)

##############################################################################################################################################################################
##############################################################################################################################################################################

##############################################################################################################################################################################
##############################################################################################################################################################################

tensor_bus_population = read.csv("output/tensor_output_bus_gpu_only.csv")
tensor_bus_population$time_seconds = tensor_bus_population$time_seconds / (tensor_bus_population$generations / 1000)
tensor_bus_population = aggregate(time_seconds ~ population + device, tensor_bus_population, mean)
cpu_population_subset = subset(tensor_bus_population, device == "cpu")
cpu_other_population_subset = subset(tensor_bus_population, device == "cpu_impl")
gpu_population_subset = subset(tensor_bus_population, device == "cuda")

tensor_bus_population_frame_cpu_other = data.frame(
  population = cpu_other_population_subset$population,
  time_cpu_other = cpu_other_population_subset$time_seconds
)

tensor_bus_population_frame_cpu = data.frame(
  population = cpu_population_subset$population,
  time_cpu = cpu_population_subset$time_seconds
)

tensor_bus_population_frame_gpu = data.frame(
  population = gpu_population_subset$population,
  time_gpu = gpu_population_subset$time_seconds
)

breaks = c(25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800)

tensor_bus_population_plot = ggplot() + 
  geom_line(data = tensor_bus_population_frame_cpu_other, aes(x = population, y = time_cpu_other, colour = "CPU Java"), linewidth = 1.25) +
  geom_point(data = tensor_bus_population_frame_cpu_other, aes(x = population, y = time_cpu_other, colour = "CPU Java"), size = 3) +
  geom_line(data = tensor_bus_population_frame_cpu, aes(x = population, y = time_cpu, colour = "CPU Tensor"), linewidth = 1.25) +
  geom_point(data = tensor_bus_population_frame_cpu, aes(x = population, y = time_cpu, colour = "CPU Tensor"), size = 3) +
  geom_line(data = tensor_bus_population_frame_gpu, aes(x = population, y = time_gpu, colour = "GPU Tensor"), linewidth = 1.25) +
  geom_point(data = tensor_bus_population_frame_gpu, aes(x = population, y = time_gpu, colour = "GPU Tensor"), size = 3) +
  scale_y_log10(breaks = breaks) +
  scale_x_log10(breaks = c(500, 1000, 1500, 2000, 3000, 5000, 10000, 20000)) +
  labs(title = "\nČasová závislosť od populácie pre rôzne implementácie GA pre úlohu č. 2 - logaritmická mierka\n", x = "Populácia", y = "Čas (s)", color = "Implementácia") +
  scale_color_manual(values=c("#CC66FF", "#CC6666", "#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

tensor_bus_population_plot
ggsave(file="output/img/tensor_bus_population.svg", plot=tensor_bus_population_plot, width=10, height=8)

#############################################################################################################################################################################

tensor_bus_population_plot_nolog = ggplot() + 
  geom_line(data = tensor_bus_population_frame_cpu_other, aes(x = population, y = time_cpu_other, colour = "CPU Java"), linewidth = 1.25) +
  geom_point(data = tensor_bus_population_frame_cpu_other, aes(x = population, y = time_cpu_other, colour = "CPU Java"), size = 3) +
  geom_line(data = tensor_bus_population_frame_cpu, aes(x = population, y = time_cpu, colour = "CPU Tensor"), linewidth = 1.25) +
  geom_point(data = tensor_bus_population_frame_cpu, aes(x = population, y = time_cpu, colour = "CPU Tensor"), size = 3) +
  geom_line(data = tensor_bus_population_frame_gpu, aes(x = population, y = time_gpu, colour = "GPU Tensor"), linewidth = 1.25) +
  geom_point(data = tensor_bus_population_frame_gpu, aes(x = population, y = time_gpu, colour = "GPU Tensor"), size = 3) +
  #scale_y_log10(breaks = breaks) +
  #scale_x_log10(breaks = c(500, 1000, 1500, 2000, 3000, 5000, 10000, 20000)) +
  labs(title = "\nČasová závislosť od populácie pre rôzne implementácie GA pre úlohu č. 2\n", x = "Populácia", y = "Čas (s)", color = "Implementácia") +
  scale_color_manual(values=c("#CC66FF", "#CC6666", "#9999CC")) +
  theme_minimal() +
  theme(legend.position = "right", legend.direction = "vertical", legend.title = element_text(face = "bold", size = 15))

tensor_bus_population_plot_nolog
ggsave(file="output/img/tensor_bus_population_nolog_1.svg", plot=tensor_bus_population_plot_nolog, width=10, height=8)

##############################################################################################################################################################################
