

while True:
    utterance = input('> ')
    if len(utterance) == 0:
        exit_ut = input('Exit? [yes]/no > ')
        if (len(exit_ut) == 0) | (exit_ut.lower() == 'yes'):
            break
    response = 'good.'
    print(response)
