# Base Architecture Extraction from Carlini

## Overview of structure

To evaluate this on a cluster, be sure to clone this repository

```git clone https://github.com/pchaitanya21/base_extraction_implementaion.git```

```cd base_architecture_extraction```

then install dependencies

```pip install -r requirements.txt```

you may need to create a job script, but here is an example of how to make it run for the perplexity/extraction

```python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path monology/pile-uncopyrighted```

## For Running Job Script on the Eddie Cluster
(Think the cirus uses slurm so just has slightly different .sh file and submit through sbatch)

### ssh into the cluster: 

``` ssh s1234567@eddie.ecdf.ed.ac.uk``` 

### navigate to your scratch space

```cd /exports/eddie/scratch/s1234567```

### create/overwrite job script

#### Nano 
```nano job_script.sh```

paste/edit the script (sample script is included as eddie_job.sh)

### Save & Exit

Press `Ctrl + O` to save the file.
Press `Enter` to confirm the file name.
Press `Ctrl + X` to exit nano


#### Vim

```vim job_script.sh```

insert/make changes 
`i`

Normal mode `esc` 

Save & Exit:  `:wq` and press `Enter`
Force Quit w/o saving: `:q!`

(if you cannot exit try clickig `esc` first to put you in normal mode)

### Submit the Script 

```qsub job_script.sh```

```qsub -l h_rt=H:M:S jobscript.sh``` (instead of including the limitations in your script you can also just include them on the command line instead)

```sbatch cirus_script.sh```

### Copying a file from the cluster to local 

Sample, make sure that you are **NOT** logged into SSH. If you are open another locally or logout and then try. 
```scp <youruun>@eddie.ecdf.ed.ac.uk:myfile.txt /some/local/directory```

Ex: 
```scp s2558433@eddie.ecdf.ed.ac.uk:/exports/eddie/scratch/s2558433/base_extraction_implementaion/output_scores_EleutherAI_pythia-2.8b_EleutherAI_pythia-160m.csv /Users/deals/Desktop```

### More ways for Transfering data to and from Eddie
Found [here](https://www.geos.ed.ac.uk/~smudd/LSDTT_docs/html/edin_instructions.html), it has more documentation for running scripts/all sorts of commands for Eddie Cluster

You need to copy all the files needed to run your job on Eddie. To copy to and from the cluster, you need to use the secure copy command scp. Some examples of syntax are given below:

Copy the file “myfile.txt” from your computer (the local host) to Eddie (the remote host):

```scp myfile.txt <youruun>@eddie.ecdf.ed.ac.uk:/some/remote/directory```
Copy the file “myfile.txt” from Eddie to the local host:

```scp <youruun>@eddie.ecdf.ed.ac.uk:myfile.txt /some/local/directory```
Copy the directory “mydir” from the local host to a remote host’s directory “fardir”:

```scp -r mydir <youruun>@eddie.ecdf.ed.ac.uk:/some/remote/directory/fardir```
Copy multiple files from the remote host to your current directory on the local host:

```scp <youruun>@eddie.ecdf.ed.ac.uk:~/\{myfile1.txt,myfile2.txt\}```

 .
More examples can be found at http://www.hypexr.org/linux_scp_help.php. If a command is not working on Eddie, try it on the local host instead.



# TODO
- prefix and suffix

- what is discoverable memorisation?
- masked token thing for important information
- 