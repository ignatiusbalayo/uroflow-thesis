from enum import Enum

class Device(Enum):
    """Maintains the devices that where used in sampling"""
    UM = "um"
    OPPO = "opp"
    PHONE = "phone"

    @classmethod
    def get_device_names(cls):
        """Returns device names"""
        return [device.
        name for device in cls]
    
    @classmethod
    def get_devices(cls):
        """Returns the devices"""
        return list(cls)
    

if __name__ == '__main__':
    device = Device.UM
    print(Device.get_device_names())
    print(Device.get_devices())
    print(Device.UM.value)