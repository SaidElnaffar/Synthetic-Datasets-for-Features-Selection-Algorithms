import PySimpleGUI as sg

# Create a list of options for the radio buttons
options = ['Option 1', 'Option 2', 'Option 3']



'''
    layout =[
        [sg.Text("Course Name", size=(15,1)), sg.InputText(courseName,key='-courseName-') ],
        [sg.Text('CSV File', size=(15,1)), sg.InputText(csvFile, key='-csvFile-', enable_events=True), sg.FileBrowse(file_types=(("CSV Files", "*.CSV"),))],
        [sg.Text('Start at column', size=(15,1), key='-column-'), sg.InputText(starting_column) ],
        checkboxes_list,
        [sg.Button("Upload"), sg.Cancel()]
    ]

'''



# Create a layout with Radio elements
layout = [[sg.Radio(option, 1, key=option) for option in options]]

# Create the window
window = sg.Window('Radio Buttons', layout)

# Event loop to process events
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

# Clean up
window.close()
