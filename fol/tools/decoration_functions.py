from datetime import datetime
import time
import functools

def print_with_timestamp_and_execution_time(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        class_name = self.__class__.__name__
        object_name = self.GetName()
        
        print(f"{current_time} - Info : {object_name}.{func.__name__} finished in {execution_time:.4f} seconds")
        return result
    return wrapper