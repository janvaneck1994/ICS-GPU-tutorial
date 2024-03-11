# ICS GPU Tutorial on CS01

Follow these instructions to set up and test a GPU environment for the ICS GPU tutorial on CS01.

## Step 1: Access Your Scratch Directory

Navigate to your scratch directory on CS01:

```bash
cd /scratch/<solis-id>
```

Replace `<solis-id>` with your actual SOLIS ID.

## Step 2: Clone the Tutorial Repository

Clone the ICS GPU tutorial repository from GitHub:

```bash
git clone https://github.com/janvaneck1994/ICS-GPU-tutorial.git
```

Make it your working dir

```bash
cd ICS-GPU-tutorial
```

## Step 3: Load Python Module

Load the Python module on CS01:

```bash
module load python
```

## Step 4: Verify Python Version

Check if the correct version of Python is loaded:

```bash
which python
```

You should see the output similar to:

```
alias python='python3.9'
/usr/bin/python3.9
```

## Step 5: Create a Python Virtual Environment

Create a new Python virtual environment for the tutorial:

```bash
python -m venv ics_gpu_tutorial
```

## Step 6: Activate the Virtual Environment

Activate the created virtual environment:

```bash
source ics_gpu_tutorial/bin/activate
```

## Step 7: Verify Python Version in Virtual Environment

Check if the correct version of Python is being used in the virtual environment:

```bash
which python
```

You should see the output similar to:

```
alias python='python3.9'
/storage/scratch/<solis-id>/ICS-GPU-tutorial/ics_gpu_tutorial/bin/python3.9
```

Ensure you replace `<solis-id>` with your actual SOLIS ID.

## Step 8: Install PyTorch

Install PyTorch and torchvision in your newly created venv using pip:

```bash
pip install torch torchvision
```

## Step 9: Submit the GPU Test Job

Submit the test job to SLURM. This job will check if the gpu is available from pytorch:

```bash
sbatch test_gpu.sh
```

## Step 10: Check the Test Output

After the job has succeeded, check the output file `test_gpu.out`:

You should see:

```
GPU is available.
```

This confirms that the GPU is properly configured and available for use.

---

Replace `<solis-id>` with your actual SOLIS ID where applicable. This tutorial guides you through setting up a Python environment for GPU computing on CS01.
