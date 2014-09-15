def create_text(command):
    text = ""
    text += "#!/bin/sh\n"
    text += "#$ -cwd              # Set the working directory for the job to the current directory\n"
    text += "#$ -V\n"
    text += "#$ -l h_rt=24:0:0    # Request 24 hour runtime\n"
    text += "#$ -l h_vmem=1.5G      # Request 256MB RAM\n"
    text += "{0}".format(command)
    return text

for pop_size in [1000,1500,2000,2500,3000]:
    for lim_percentage in [10,20]:
        for num_epochs in [50]:
            for lr in [0.05]:
                for hiddens in [10,20,50,100]:
                    for rtr in [1]:
                        for w in [0.1,0.05,0.2]:
                            w = int(pop_size*w)
                            for trial in range(0,3):
                                command = "python2.6 daga.py {0} {1} {2} {3} {4} {5} {6} {7}".format(
                                    pop_size,lim_percentage,num_epochs,lr,hiddens,rtr,w,trial)
                                f_text = create_text(command)
                                f = open("ae_hiff_128_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.job".format(pop_size,lim_percentage,num_epochs,lr,hiddens,rtr,w,trial),"w")
                                f.write(f_text)

