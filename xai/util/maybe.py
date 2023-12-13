from typing import *
from abc import ABC, abstractmethod
from dataclasses import dataclass

X = TypeVar("X")
Y = TypeVar("Y")

class Maybe(Generic[X], ABC):

    @abstractmethod
    def is_some(self) -> bool:
        pass

    @abstractmethod
    def is_nil(self) -> bool:
        pass

    @abstractmethod
    def is_error(self) -> bool:
        pass
    
    @abstractmethod
    def get(self) -> X|NoReturn:
        pass

    @abstractmethod
    def get_or_else(self, default: Callable[[],X]) -> X:
        pass

    @abstractmethod
    def or_else(self, default: Callable[[],"Maybe[X]"]) -> "Maybe[X]":
        pass

    @abstractmethod
    def flatmap(self, transform: Callable[[X],"Maybe[Y]"]) -> "Maybe[Y]":
        pass

    @abstractmethod
    def filter(self, predicate: Callable[[X],bool]) -> "Maybe[X]":
        pass
        
    def map(self, transform: Callable[[X],Y]) -> "Maybe[Y]":
        return self.flatmap(lambda x: eval(lambda: transform(x)))
    
    def foreach(self, execution: Callable[[X],None]) -> None:
        self.map(lambda x: execution(x))

    @abstractmethod
    def __iter__(self) -> Iterator[X]:
        pass
        
class Try(Maybe[X]):
    pass

class Option(Maybe[X]):
    pass

@dataclass
class Some(Try[X], Option[X]):
    value: X

    def is_some(self) -> Literal[True]:
        return True
    
    def is_nil(self) -> Literal[False]:
        return False

    def is_error(self) -> Literal[False]:
        return False
    
    def get(self) -> X:
        return self.value

    def get_or_else(self, _: Callable[[],X]) -> X:
        return self.value
    
    def or_else(self, _: Callable[[],Maybe[X]]) -> Maybe[X]:
        return self
    
    def flatmap(self, transform: Callable[[X],Maybe[Y]]) -> Maybe[Y]:
        return transform(self.value)
    
    def filter(self, predicate: Callable[[X],bool]) -> Maybe[X]:
        if predicate(self.value):
            return Some(self.value)
        else:
            return Nil()
        
    def __iter__(self) -> Iterator[X]:
        return iter((self.get(),))

@dataclass
class Error(Try[X]):
    exception: Exception

    def is_some(self) -> Literal[False]:
        return False
    
    def is_nil(self) -> Literal[False]:
        return False

    def is_error(self) -> Literal[True]:
        return True
    
    def get(self) -> NoReturn:
        raise self.exception

    def get_or_else(self, default: Callable[[],X]) -> X:
        return default()
    
    def or_else(self, default: Callable[[],Maybe[X]]) -> Maybe[X]:
        return default()
    
    def flatmap(self, _: Callable[[X],Maybe[Y]]) -> "Error[Y]":
        return Error(self.exception)
    
    def filter(self, _: Callable[[X], bool]) -> "Error[X]":
        return Error(self.exception)
    
    def __iter__(self) -> Iterator[X]:
        return iter(tuple())
    
class Nil(Option[X]):
    
    def is_some(self) -> Literal[False]:
        return False
    
    def is_nil(self) -> Literal[True]:
        return True

    def is_error(self) -> Literal[False]:
        return False
    
    def get(self) -> NoReturn:
        raise TypeError()

    def get_or_else(self, default: Callable[[],X]) -> X:
        return default()
    
    def or_else(self, default: Callable[[],"Maybe[X]"]) -> "Maybe[X]":
        return default()
    
    def flatmap(self, _: Callable[[X],Maybe[Y]]) -> "Nil[Y]":
        return Nil()
    
    def filter(self, _: Callable[[X], bool]) -> "Nil[X]":
        return Nil()
    
    def __iter__(self) -> Iterator[X]:
        return iter(tuple())
    
def some(value: X) -> Some[X]:
    return Some(value)

def nil() -> Nil[X]:
    return Nil()

def error(exception: Exception) -> Error[X]:
    return Error(exception)

def eval(expression: Callable[[],X]) -> Maybe[X]:
    try:
        result = expression()
        if result:
            return Some(result)
        else:
            return Nil()
    except Exception as exception:
        return Error(exception)