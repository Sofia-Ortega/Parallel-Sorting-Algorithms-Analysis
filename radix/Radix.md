
## Running

1. Run `. build.sh`

2. Run sbatch:

```
sbatch radix.grace_job <input_file> <n> <p>
```

- `input_file`: file that contains one number per line to be sorted
- `n`: number of numbers found in input_file
- `p`: number of processes