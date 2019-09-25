import foundations

NUM_JOBS = 100
from constants import generate_config

for job_num in range(NUM_JOBS):
    print(f'job number {job_num}')
    config_dict = generate_config()
    print('Finished writing config.yml file')

    foundations.submit(
        scheduler_config="scheduler",
        command=["main.py"],
        project_name="Fake-Audio-Detection",
    )
