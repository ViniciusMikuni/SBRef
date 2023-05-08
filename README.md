# Schrodinger Bridges for Simulation Refinement

From NERSC, you can load the tensorflow module with
```bash
module load tensorflow
```

# Data
To train the model using the [CaloGAN](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.97.014021) dataset you can download the files from [here](https://data.mendeley.com/datasets/pvn3xc3wy5/1)


Let's first run a baseline model using a VAE:

```bash
cd scripts
python calogan.py --model VAE 
```

The training will create a checkpoint from where you can load the VAE model as the starting point of the Schrodinger Bridge. To train the refiner use:

```bash
python calogan.py --model VAE --SB
```

After training a few iterations, you can plot the response of the refiner using:

```bash
python plot_calogan.py --model VAE --stage NSTAGE
```
where the flag ```NSTAGE``` is the IPF iteration you want to look at.