from channels.generic.websocket import AsyncJsonWebsocketConsumer


class JobConsumer(AsyncJsonWebsocketConsumer):

    async def connect(self):
        self.job_id = self.scope["url_route"]["kwargs"]["job_id"]

        self.group = f"job_{self.job_id}"

        await self.channel_layer.group_add(
            self.group,
            self.channel_name,
        )

        await self.accept()


    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group,
            self.channel_name,
        )


    async def detection_partial(self, event):
        await self.send_json(
            {
                "type": "detection.partial",
                **event,
            }
        )


    async def detection_complete(self, event):
        await self.send_json(
            {
                "type": "detection.complete",
                **event,
            }
        )


    async def detection_failed(self, event):
        await self.send_json(
            {
                "type": "detection.failed",
                **event,
            }
        )