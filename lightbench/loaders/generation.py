class Generation:
    def __init__(self, response, inferece_time=0, ttft=0, peak_memory_usage=0, avg_power_usage=0):
        self.response = response
        self.inferece_time = inferece_time
        self.ttft = ttft
        self.peak_memory_usage = peak_memory_usage
        self.avg_power_usage = avg_power_usage
