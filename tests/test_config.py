
import pytest


class NotInRangeException(Exception):
    def __init__(self, message = 'Not In Range'):
        super().__init__(message)
        
def test_generic():
    a = 10
    if a not in range(10, 20):
        raise NotInRangeException
        
def test_equal():
    a = 2
    b = 2
    assert a ==b 
    
 

