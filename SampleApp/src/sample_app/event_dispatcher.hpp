#pragma once
#include <functional>
#include <any>
#include <unordered_map>
#include <vector>
#include <string>

namespace SampleApp
{
	struct Event
	{
		explicit Event(std::string_view signal) : signal(signal) {}
		std::string signal;

		bool operator==(const Event& other) const;
	};

	// Custom hash function for Event
	struct EventHash
	{
		std::size_t operator()(const Event& event_) const;
	};

	class Listener
	{
	public:
		Listener(std::function<void(std::any)> func) : callback(std::move(func)) {}
		void invoke(const std::any& data) const;

	private:
		std::function<void(std::any)> callback;
	};

	// EventDispatcher class
	class EventDispatcher
	{
	public:
		void registerListener(const Event& event_, const Listener& listener);

		template<typename T>
		void emitSignal(const Event& event_, const T& data)
		{
			if (m_listeners.find(event_) != m_listeners.end()) {
				for (const Listener& listener : m_listeners[event_]) {
					listener.invoke(std::make_any<T>(data));
				}
			}
		}

	private:
		std::unordered_map<Event, std::vector<Listener>, EventHash> m_listeners;
	};
}