import aiomqtt
import asyncio, sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    # 1. 连上本机的 EMQX
    async with aiomqtt.Client("localhost", 1883) as client:
        # 2. 订阅频道 microduck/robot01/cmd
        await client.subscribe("microduck/robot01/cmd")
        print("正在监听")

        # 3. 监听
        async for message in client.messages:
            print(f"收到了 MQTT 消息: {message.payload.decode()}")

asyncio.run(main())