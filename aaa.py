
def main():
	global epsilon
	global log_values

	for e in range(3000):
		log_values = []
		running_rew = 0

		state = env.reset()
		state = np.reshape(state, [1, 64, 64, 1])

		for t in range(2000):

			result = gc.collect()

			print(str(t+1)+"/100")
			sys.stdout.write("\033[F")

			state = env.get_observation()
            state = np.reshape(state, [1, 64, 64, 1])

            action = act(state)

			observation, reward, done, info = env.step(action)
            observation = np.reshape(observation, [1, 64, 64, 1])

			remember(state, action, reward, observation, done)

			running_rew = running_rew + reward

			state = observation

			if done or t == 500:
				print("episode: {}, score: {}, eps: {}, memory: {}, collisions: {}".format(e, running_rew, epsilon, len(memory), env.collisions))
				break

			if len(memory) > 32:
				replay(32)

		update_weights()
		model.save_weights("weights/"+str(e)+'_my_model_weights.h5')

		with open('moves/'+str(e) +'_file.csv', mode='w') as moves_file:
				employee_writer = csv.writer(moves_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				employee_writer.writerow(log_values)


if __name__=="__main__":
    model = create_cnn()
    target_model = create_cnn()
    main()
