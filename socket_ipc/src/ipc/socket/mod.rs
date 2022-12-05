use anyhow::Context;
use std::collections::HashSet;
use std::os::unix::net::{UnixListener,UnixStream};
use std::io::{Read, Write};
use std::thread::JoinHandle;
use serde::{Deserialize, Serialize};


pub trait Handler
{
    fn process(&self, bytes: Vec<u8>) -> anyhow::Result<Vec<u8>>;
}

pub fn spawn_socket_service(socket_path: &str, handler: Box<dyn Handler + Send>) -> JoinHandle<()>
{
    let socket_path = socket_path.to_owned();
    std::thread::spawn(move || {
        loop {
            if std::fs::metadata(&socket_path).is_ok() {
                //println!("A socket is already present. Deleting...");
                std::fs::remove_file(&socket_path)
                    .with_context(|| {
                        format!("could not delete previous socket at {:?}", &socket_path)
                    })
                    .unwrap();
            }

            let unix_listener = UnixListener::bind(&socket_path)
                .context("Could not create the unix socket")
                .unwrap();

            loop {
                let (unix_stream, _socket_address) = unix_listener
                    .accept()
                    .context("Failed at accepting a connection on the unix listener")
                    .unwrap();

                handle_stream(unix_stream, &handler).unwrap();
            }
        }
    })
}

fn handle_stream(mut unix_stream: UnixStream, handler: &Box<dyn Handler + Send>) -> anyhow::Result<()>
{
    let encoded: Vec<u8> = handler.process(get_bytes_from_stream(&mut unix_stream)?)?;

    unix_stream
        .write(&encoded[..])
        .context("Failed at writing onto the unix stream")?;

    Ok(())
}

pub fn client_send_request<T,S>(
    socket_path: &str,
    request: T,
) -> anyhow::Result<S>
    where
        T: Serialize,
        Vec<u8>: TryFrom<T>,
        S: for<'a> Deserialize<'a> + TryFrom<Vec<u8>>,

{
    let mut unix_stream = UnixStream::connect(socket_path).context("Could not create stream")?;

    write_request_and_shutdown(&mut unix_stream, request.try_into().map_err(|_| anyhow::anyhow!("try_into() failed"))?)?;
    Ok(get_bytes_from_stream(&mut unix_stream)?.try_into().map_err(|_| anyhow::anyhow!("try_into() failed"))?)
}

fn write_request_and_shutdown(
    unix_stream: &mut UnixStream,
    request: Vec<u8>,
) -> anyhow::Result<()> {
    unix_stream
        .write(&request)
        .context("Failed at writing onto the unix stream")?;

    unix_stream
        .shutdown(std::net::Shutdown::Write)
        .context("Could not shutdown writing on the stream")?;
    Ok(())
}

fn get_bytes_from_stream(unix_stream: &mut UnixStream) -> anyhow::Result<Vec<u8>> {
    let mut bytes: Vec<u8> = Vec::new();
    unix_stream
        .read_to_end(&mut bytes)
        .context("Failed at reading the unix stream")?;
    Ok(bytes)
}