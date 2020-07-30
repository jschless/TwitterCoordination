class Tweet(object):
    def __init__(self, dic):
        for key, value in dic.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        return f"{self.text} - {self.username}"
    
class MissingTweet(object):
    def __init__(self, id, username):
        self.id = id
        self.username = username
        self.text = 'Missing'
