from locust import HttpUser, task, between, events

PATH_TO_CSV_FILE=""
stat_file = open(PATH_TO_CSV_FILE, 'w+')

class ApiUser(HttpUser):
  wait_time = between(1, 5)

  @task
  def en_mr(self):
    
    URL=""

    data={"sourceLanguage": "en", "targetLanguage": "mr"}

    PATH_TO_AUDIO_FILE=""
    file = open(PATH_TO_AUDIO_FILE, 'rb')

    self.client.post(url=URL, files={"files": file}, data=data)


# hook that is fired each time the request ends up with success
@events.request_success.add_listener
def hook_request_success(response_time, **kw):
    stat_file.write(str(response_time) + "\n")


@events.quitting.add_listener
def hook_quitting(environment, **kw):
    stat_file.close()