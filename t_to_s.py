import win32com.client 
speaker = win32com.client.Dispatch("SAPI.SpVoice") 
class Hero():
    def __init__(self, inpu):
        self.inpu = inpu
    def Inputss(self):
        if (self.inpu != "no gesture" and self.inpu != "doing other things"):
            speaker.Speak(self.inpu)
           

# ttos = Hero(final_label)
# ttos.Inputss()
