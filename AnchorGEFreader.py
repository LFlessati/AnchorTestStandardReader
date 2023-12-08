import numpy as np
import math
import matplotlib.pyplot as plt



######################
# Reading the gef file
######################
NAME = 'Example_results'
name_file = NAME + '.gef'



##########################
# Definition of the output
##########################
plot_figure_phases = 'Yes' #Plotting the figures of each phase: Yes or No
save_figure_phases = 'Yes' #Saving the figures of each phase: Yes or No
plot_figure_creep  = 'Yes' #Plotting the figure for creep: Yes or No
save_figure_creep  = 'Yes' #Saving the figure for creep: Yes or No
save_data_creep    = 'Yes' #Saving the data of the creep: Yes or No


################
# Initialization
################

#Reading the data:
data_array = np.loadtxt(name_file, delimiter=';', dtype=str)

#Removing void data
valid_rows = data_array[:, 0] != '-9999'
valid_rows = data_array[:, 1] != '-9999'
valid_rows = data_array[:, 2] != '-9999'
filtered_data = data_array[valid_rows]
filtered_data[:, 2] = np.char.rstrip(filtered_data[:, 2], '!')

#Putting the read data inro arrays:
time   = filtered_data[:,0].astype(float)
load   = filtered_data[:,1].astype(float)
disp   = filtered_data[:,2].astype(float)

#Detection of the different load phases:
n_load_phases = 0
indices = np.array(np.where(time == 1))[0]
N_phases = indices.size

#Initialization of vectors for creep:
Load_val_plot = np.zeros((N_phases-1,1))
Ks_plot       = np.zeros((N_phases-1,1))
c_plot        = np.zeros((N_phases-1,1))


#####################
# Test interpretation
#####################

for i in range(0,N_phases-1):
    #linear interpolation of the log(time) vs displacement
    pr_time = time[indices[i]:indices[i+1]-1]
    log_time = np.log10(pr_time)
    y = disp[indices[i]:indices[i+1]-1]
    A = np.vstack([log_time, np.ones(len(log_time))]).T
    ks, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    #saving the data for the creep-load results
    c_plot[i] = c
    Ks_plot[i] = ks
    Load_val_plot[i] = load[indices[i]]
    
if save_data_creep == 'Yes':
    header = "Load [kN] Creep [mm] :"
    np.savetxt(NAME + '_Creep_data.txt', np.column_stack(( Load_val_plot,Ks_plot)), header=header) 
    


######################
# Plotting the results
######################

#closing eventual figure that are open
plt.close('all')    

#Various test phases:
if plot_figure_phases == 'Yes' or save_figure_phases == 'Yes':
    for i in range(0,N_phases-1):
        plt.figure(i+2,figsize=(15,5))
        
        plt.subplot(131)
        plt.plot(time[indices[i]-1:indices[i+1]-1], load[indices[i]-1:indices[i+1]-1])
        plt.xlabel('Time [min]')
        plt.ylabel('Load [kN]')
        plt.title('Time vs Load')
        
        plt.subplot(132)
        plt.plot(disp[indices[i]-1:indices[i+1]-1], load[indices[i]-1:indices[i+1]-1])
        plt.xlabel('Displacement [mm]')
        plt.ylabel('Load [kN]')
        plt.title('Displacement vs Load')
        
        plt.subplot(133)
        plt.semilogx(time[indices[i]-1:indices[i+1]-1], disp[indices[i]-1:indices[i+1]-1],marker='o',label='Experimental data')
        plt.xlabel('Time [min]')
        plt.ylabel('Displacement [mm]')
        plt.title('Time vs Displacement')
        
    
        pr_time = time[indices[i]:indices[i+1]-1]
        pr_dis = c_plot[i] +  Ks_plot[i]*np.log10(pr_time)
        
        plt.semilogx(pr_time,pr_dis,label='Linear interpolation')
        plt.legend() 
        
        if save_figure_phases == 'Yes':
            plt.savefig(NAME + f'_Phase_{i+1}' + '.png')   

#Creep vs load:
if plot_figure_creep == 'Yes' or save_figure_creep == 'Yes':
    plt.figure(1,figsize=(5,5))
    plt.scatter(Load_val_plot,Ks_plot)
    plt.xlabel('Load [kN]')
    plt.ylabel('Creep [mm]')
    plt.title('Load vs Creep')

    if save_figure_creep == 'Yes':
        plt.savefig(NAME + '_Creep' + '.png')  

