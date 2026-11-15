# NAB replay (first 15% is Phase I, 12-point alarm cooldown)

| Stream                             | Chart    | Prewhiten | Windows hit | False alarms | Per 1k | Delay |
| ---------------------------------- | -------- | --------- | ----------- | ------------ | ------ | ----- |
| nyc_taxi                           | shewhart | no        | 1/5         | 15           | 1.71   | 159   |
| nyc_taxi                           | shewhart | yes       | 5/5         | 126          | 14.36  | 47    |
| nyc_taxi                           | ewma     | no        | 3/5         | 22           | 2.51   | 130.3 |
| nyc_taxi                           | ewma     | yes       | 5/5         | 64           | 7.3    | 66    |
| nyc_taxi                           | cusum    | no        | 4/5         | 21           | 2.39   | 127.8 |
| nyc_taxi                           | cusum    | yes       | 5/5         | 77           | 8.78   | 57    |
| ambient_temperature_system_failure | shewhart | no        | 0/2         | 20           | 3.24   | -     |
| ambient_temperature_system_failure | shewhart | yes       | 2/2         | 17           | 2.75   | 184.5 |
| ambient_temperature_system_failure | ewma     | no        | 1/2         | 24           | 3.89   | 84    |
| ambient_temperature_system_failure | ewma     | yes       | 1/2         | 7            | 1.13   | 172   |
| ambient_temperature_system_failure | cusum    | no        | 1/2         | 21           | 3.4    | 293   |
| ambient_temperature_system_failure | cusum    | yes       | 1/2         | 7            | 1.13   | 163   |
| ec2_cpu_utilization_24ae8d         | shewhart | no        | 2/2         | 50           | 14.59  | 12    |
| ec2_cpu_utilization_24ae8d         | shewhart | yes       | 2/2         | 48           | 14     | 23.5  |
| ec2_cpu_utilization_24ae8d         | ewma     | no        | 2/2         | 11           | 3.21   | 94.5  |
| ec2_cpu_utilization_24ae8d         | ewma     | yes       | 2/2         | 11           | 3.21   | 94.5  |
| ec2_cpu_utilization_24ae8d         | cusum    | no        | 2/2         | 11           | 3.21   | 94.5  |
| ec2_cpu_utilization_24ae8d         | cusum    | yes       | 2/2         | 11           | 3.21   | 94.5  |
| art_daily_jumpsdown                | shewhart | no        | 1/1         | 1            | 0.29   | 266   |
| art_daily_jumpsdown                | shewhart | yes       | 1/1         | 30           | 8.75   | 201   |
| art_daily_jumpsdown                | ewma     | no        | 1/1         | 1            | 0.29   | 227   |
| art_daily_jumpsdown                | ewma     | yes       | 1/1         | 25           | 7.29   | 201   |
| art_daily_jumpsdown                | cusum    | no        | 1/1         | 1            | 0.29   | 224   |
| art_daily_jumpsdown                | cusum    | yes       | 1/1         | 40           | 11.67  | 201   |
