from datetime import datetime
import time
import functools
import inspect
import warnings

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
        
        print(f"{current_time} - Info : {class_name}.{func.__name__} - finished in {execution_time:.4f} seconds")
        return result
    return wrapper

def fol_info(message):
    # Get the current time and date
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Get the caller's frame
    caller_frame = inspect.stack()[1]
    frame = caller_frame.frame
    
    # Get the caller's class and function name
    caller_class = None
    if 'self' in frame.f_locals:
        caller_class = frame.f_locals['self'].__class__.__name__
    caller_function = caller_frame.function
    
    # Print the message with the required information
    if caller_class:
        print(f"{current_time} - Info : {caller_class}.{caller_function} - {message}")
    else:
        print(f"{current_time} - Info : {caller_function} - {message}")

def fol_error(message):
    # Get the current time and date
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Get the caller's frame
    caller_frame = inspect.stack()[1]
    frame = caller_frame.frame
    
    # Get the caller's class and function name
    caller_class = None
    if 'self' in frame.f_locals:
        caller_class = frame.f_locals['self'].__class__.__name__
    caller_function = caller_frame.function
    
    # Print the error message with the required information and stop execution
    if caller_class:
        print(f"{current_time} - Error : {caller_class}.{caller_function} - {message}")
    else:
        print(f"{current_time} - Error : {caller_function} - {message}")
    
    # Stop the execution
    raise SystemExit

def fol_warning(message):
    # Get the current time and date
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Get the caller's frame
    caller_frame = inspect.stack()[1]
    frame = caller_frame.frame
    
    # Get the caller's class and function name
    caller_class = None
    if 'self' in frame.f_locals:
        caller_class = frame.f_locals['self'].__class__.__name__
    caller_function = caller_frame.function
    
    # Print the error message with the required information and stop execution
    if caller_class:
        warning_message = f"{current_time} - Warning : {caller_class}.{caller_function} - {message}"
    else:
        warning_message = f"{current_time} - Warning : {caller_function} - {message}"
    
    # Issue the warning with stacklevel=2 to point to the caller function
    warnings.warn(warning_message, UserWarning, stacklevel=2)