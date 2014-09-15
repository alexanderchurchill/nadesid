import os,sys,time
count = 0
for pop_size in [1000,1500,2000,2500,3000]:
    for lim_percentage in [10,20]:
        for num_epochs in [50]:
            for lr in [0.05]:
                for hiddens in [10,20,50,100]:
                    for rtr in [1]:
                        for w in [0.1,0.05,0.2]:
                            w = int(pop_size*w)
                            for trial in range(0,3):
                                os.system('qsub ae_hiff_128_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.job'.format(pop_size,lim_percentage,num_epochs,lr,hiddens,rtr,w,trial))
                                time.sleep(0.1)
                                count += 1
                                print "added:",count

