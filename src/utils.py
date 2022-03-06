class Queue:
    def __init__(self, size) -> None:
        self.list = []
        self.size = size
    
    def add(self, data):
        if len(self.list) >= self.size:
            return
        self.list.append(data)

    def poll(self):
        if len(self.list) == 0:
            return 
        self.list.pop(0)
        return
    
    def peek(self):
        if len(self.list) == 0:
            return None
        return self.list[0]

    def get_len(self):
        return len(self.list)
