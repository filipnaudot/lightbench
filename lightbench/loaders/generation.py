class Generation:
    def __init__(self, response, inference_time=0, ttft=0, peak_memory_usage=0, avg_power_usage=0):
        self.response = response
        self.inference_time = inference_time
        self.ttft = ttft
        self.peak_memory_usage = peak_memory_usage
        self.avg_power_usage = avg_power_usage
