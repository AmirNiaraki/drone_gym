from tensorboardX import SummaryWriter


class Monitor:
    def __init__(self):
        self.parameters = {}

        # initialize dictionary
        self.parameters['train_step'] = 0
        self.parameters['ep_step'] = 0

        self.writer = SummaryWriter()

    def log(self, name, value, step_name):
        self.writer.add_scalar(name, value, self.parameters[step_name])
        self.writer.flush()

    def init_step_val(self, step_name):
        self.parameters[step_name] = 0

    def update_step(self, step_name):
        self.parameters[step_name] += 1

    def get(self, name):
        return self.parameters[name]

    def set(self, name, value):
        self.parameters[name] = value

    def stop(self):
        self.writer.close()
